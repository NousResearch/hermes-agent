'use strict'

// 知识源持久化 + 增量同步(阶段④)。一条「知识源」= 用户加进本机知识库的一个**文件夹或文件**,
// 连同它对应的 langflow KB,以及一份指纹 manifest(每个文件 相对路径→{mtime,size}),用于同步时 diff。
// 落盘:<HERMES_HOME>/knowledge_sources.json(纯本地,不上云;只有 catalog 摘要才经 org 管道上报)。
//
// 同步(V1)策略 —— langflow KB 只能"追加"或"删整库",没有按单文件删 chunk:
//   - 无变化            → no-op
//   - 纯新增文件        → 只灌新文件(append,便宜)
//   - 有改动/删除文件   → 整库重建(删 KB + 全量重灌;embedding 便宜,可接受)

const fs = require('node:fs')
const path = require('node:path')

const ingest = require('./knowledge-ingest.cjs')

function sourcesFilePath(home) {
  return path.join(home, 'knowledge_sources.json')
}

function readSources(home) {
  try {
    const arr = JSON.parse(fs.readFileSync(sourcesFilePath(home), 'utf-8'))
    return Array.isArray(arr) ? arr : []
  } catch {
    return []
  }
}

function writeSources(home, list) {
  try {
    fs.mkdirSync(home, { recursive: true })
    fs.writeFileSync(sourcesFilePath(home), JSON.stringify(list, null, 2))
    return true
  } catch {
    return false
  }
}

// 对渲染端隐藏 manifest(可能很大且无用),补一个 fileCount。
function publicSource(s) {
  if (!s) {
    return null
  }
  const fileCount = s.manifest ? Object.keys(s.manifest).length : (s.indexed || 0) + (s.nameOnly || 0)
  return {
    sourceId: s.sourceId,
    kb: s.kb,
    type: s.type,
    path: s.path,
    name: s.name,
    indexed: s.indexed || 0,
    nameOnly: s.nameOnly || 0,
    fileCount,
    truncated: !!s.truncated,
    lastSyncedTs: s.lastSyncedTs || 0
  }
}

function listSources(home) {
  return readSources(home).map(publicSource)
}

function getSource(home, sourceId) {
  return readSources(home).find(s => s.sourceId === sourceId) || null
}

function upsertSource(home, record) {
  const list = readSources(home)
  const i = list.findIndex(s => s.sourceId === record.sourceId)
  if (i >= 0) {
    list[i] = record
  } else {
    list.push(record)
  }
  writeSources(home, list)
  return publicSource(record)
}

// 从一次 ingestSource 的结果造一条记录。
function makeRecord(srcPath, name, result, now) {
  return {
    sourceId: result.kb,
    kb: result.kb,
    type: result.type || 'folder',
    path: srcPath,
    name,
    indexed: result.indexed || 0,
    nameOnly: result.nameOnly || 0,
    truncated: !!result.truncated,
    manifest: result.manifest || {},
    lastSyncedTs: now
  }
}

// 删一条源:删 langflow KB(best-effort)+ 删记录。
async function removeSource(home, sourceId, base) {
  const rec = getSource(home, sourceId)
  if (rec && base) {
    try {
      await ingest.deleteKb(base, rec.kb)
    } catch {
      /* KB 删不掉也继续删记录 */
    }
  }
  const list = readSources(home)
  const next = list.filter(s => s.sourceId !== sourceId)
  writeSources(home, next)
  return { ok: true, removed: next.length !== list.length }
}

// diff 两份 manifest → 新增 / 改动 / 删除(都用相对路径 key)。
function diffManifest(oldM, newM) {
  oldM = oldM || {}
  newM = newM || {}
  const added = []
  const modified = []
  const removed = []
  for (const rel of Object.keys(newM)) {
    if (!(rel in oldM)) {
      added.push(rel)
    } else if (oldM[rel].m !== newM[rel].m || oldM[rel].s !== newM[rel].s) {
      modified.push(rel)
    }
  }
  for (const rel of Object.keys(oldM)) {
    if (!(rel in newM)) {
      removed.push(rel)
    }
  }
  return { added, modified, removed }
}

// 同步一条源:重扫 → diff → 纯新增就 append、有改/删就重建。base = 已就绪的 langflow 地址。
async function syncSource(home, sourceId, base, sender, now) {
  const rec = getSource(home, sourceId)
  if (!rec) {
    return { ok: false, error: 'not-found' }
  }
  let st
  try {
    st = fs.statSync(rec.path)
  } catch {
    return { ok: false, error: 'source-missing' } // 源被删/移走了
  }
  const isFile = st.isFile()
  const { textFiles, otherNames, manifest, truncated } = await ingest.collectFromPath(rec.path, isFile)
  const { added, modified, removed } = diffManifest(rec.manifest, manifest)

  if (!added.length && !modified.length && !removed.length) {
    return { ok: true, changed: false, added: 0, modified: 0, removed: 0 }
  }

  try {
    if (modified.length || removed.length) {
      // 有改/删 → 整库重建(langflow 没有按单文件删)
      await ingest.deleteKb(base, rec.kb)
      await ingest.ensureKb(base, rec.name)
      await ingest.ingestInto(base, rec.kb, rec.name, textFiles, otherNames, sender)
    } else {
      // 纯新增 → 只灌新文件
      const isNew = abs => added.includes(isFile ? path.basename(abs) : path.relative(rec.path, abs))
      await ingest.ingestInto(
        base,
        rec.kb,
        rec.name,
        textFiles.filter(isNew),
        otherNames.filter(rel => added.includes(rel)),
        sender
      )
    }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }

  upsertSource(home, {
    ...rec,
    manifest,
    indexed: textFiles.length,
    nameOnly: otherNames.length,
    truncated,
    lastSyncedTs: now
  })
  return { ok: true, changed: true, added: added.length, modified: modified.length, removed: removed.length }
}

module.exports = {
  listSources,
  getSource,
  upsertSource,
  makeRecord,
  removeSource,
  syncSource,
  diffManifest,
  sourcesFilePath
}
