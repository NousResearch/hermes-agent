'use strict'

// 把"确认入库"接通。在本地 langflow 建一个 KB,锁 keyless 本地 embedding(FastEmbed,正文
// 不出本机),然后:
//   - 文本类(TEXT_EXT)分批读内容 → ingest(langflow embed 正文)
//   - 其余文件把"相对路径/文件名"汇总成一个清单文本喂进去(可被检索到"有这个文件";
//     真要读内容时由上游多模态模型 GPT/Claude 处理)
// 源可以是**文件夹或单个文件**(detect via stat)。同时产出一份 manifest(每个文件的
// 相对路径 → {mtime,size}),给"增量同步"做 diff 用。进度通过 sender 回传给渲染端。

const fs = require('node:fs')
const path = require('node:path')
const { resolveDirectoryForIpc, resolveReadableFileForIpc } = require('./hardening.cjs')
const { TEXT_EXT, SKIP_DIRS, NOISE_NAMES, NOISE_EXT, MAX_TEXT_BYTES } = require('./knowledge-inventory.cjs')

const LANGFLOW_URL = 'http://127.0.0.1:7860'
// 与 langflow LOCAL_EMBEDDING_PROVIDER / LOCAL_EMBEDDING_DEFAULT 对齐
// (src/lfx/src/lfx/base/models/unified_models/{class_registry,model_catalog}.py)。
const EMBEDDING_PROVIDER = 'Kari 本地'
const EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
const INGEST_BATCH = 16
const MAX_DEPTH = 12
const MAX_FILES = 100000

function sendProgress(sender, payload) {
  try {
    sender?.send('hermes:knowledge:ingest-progress', payload)
  } catch {
    /* sender 可能已销毁 */
  }
}

// langflow 的知识库路由即便进程开了 LANGFLOW_SKIP_AUTH_AUTO_LOGIN 仍要鉴权(实测无 token →
// 403 "No authentication credentials provided"),所以调用前取一个 auto_login token 带上,
// 与 tools/expose_flow_tool 一致。取不到就裸调(真正放开鉴权的环境仍可用)。按 base 缓存。
let _lfToken = null
let _lfTokenBase = null

async function fetchAutoLoginToken(base) {
  try {
    const res = await fetch(`${base}/api/v1/auto_login`, { headers: { accept: 'application/json' } })
    if (res.ok) {
      return (await res.json().catch(() => ({})))?.access_token || null
    }
  } catch {
    /* 取不到 token 就裸调 */
  }
  return null
}

async function lfToken(base) {
  if (_lfToken && _lfTokenBase === base) {
    return _lfToken
  }
  _lfToken = await fetchAutoLoginToken(base)
  _lfTokenBase = base
  return _lfToken
}

async function lfFetch(base, pathname, options = {}) {
  const token = await lfToken(base)
  const headers = { ...(options.headers || {}) }
  if (token) {
    headers.authorization = `Bearer ${token}`
  }
  return fetch(`${base}${pathname}`, { ...options, headers })
}

// langflow 建 KB 时把名字里的空格转下划线作为目录名;复用时按同规则推断。
function kbDirName(name) {
  return name.trim().replace(/ /g, '_')
}

async function ensureKb(base, name) {
  const res = await lfFetch(base, '/api/v1/knowledge_bases', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      name,
      embedding_provider: EMBEDDING_PROVIDER,
      embedding_model: EMBEDDING_MODEL
    })
  })

  if (res.ok) {
    const info = await res.json().catch(() => ({}))
    return info.dir_name || kbDirName(name)
  }
  // 409 = 已存在,复用同名 KB(追加 ingest)。
  if (res.status === 409) {
    return kbDirName(name)
  }
  const text = await res.text().catch(() => '')
  throw new Error(`建知识库失败 HTTP ${res.status} ${text.slice(0, 200)}`)
}

// 删整个 KB —— 同步时"改/删了文件"需重建(langflow 没有按单文件删 chunk 的接口),先删后重灌。
async function deleteKb(base, kbDir) {
  try {
    const res = await lfFetch(base, `/api/v1/knowledge_bases/${encodeURIComponent(kbDir)}`, { method: 'DELETE' })
    return res.ok || res.status === 404
  } catch {
    return false
  }
}

async function ingestFiles(base, kbDir, files) {
  const form = new FormData()
  for (const f of files) {
    form.append('files', new Blob([f.buffer]), f.name)
  }
  const res = await lfFetch(base, `/api/v1/knowledge_bases/${encodeURIComponent(kbDir)}/ingest`, {
    method: 'POST',
    body: form
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`入库失败 HTTP ${res.status} ${text.slice(0, 200)}`)
  }
  return res.json().catch(() => ({}))
}

// 归类一个文件:文本类且不超大 → 可全文索引(读正文 embed);其余 → 仅文件名。
// noise 文件返回 null(跳过)。manifest 用相对路径做 key,{m: mtime_ms, s: size} 做指纹。
function classifyFile(full, relName, stat) {
  const ext = path.extname(full).toLowerCase()
  const base = path.basename(full)
  if (NOISE_NAMES.has(base) || NOISE_EXT.has(ext)) {
    return null
  }
  const size = stat ? stat.size : 0
  const fp = { m: stat ? Math.round(stat.mtimeMs) : 0, s: size }
  if (TEXT_EXT.has(ext) && size <= MAX_TEXT_BYTES) {
    return { kind: 'text', abs: full, rel: relName, fp }
  }
  return { kind: 'name', abs: full, rel: relName, fp }
}

// 扫描源(文件夹或单文件)→ { textFiles:[abs], otherNames:[rel], manifest:{rel:{m,s}}, truncated }。
// manifest 覆盖文本 + 仅文件名两类(都要参与 diff)。
async function collectFromPath(srcPath, isFile) {
  const textFiles = []
  const otherNames = []
  const manifest = {}
  let truncated = false

  if (isFile) {
    let stat = null
    try {
      stat = await fs.promises.stat(srcPath)
    } catch {
      /* stat 失败仍按仅文件名登记 */
    }
    const rel = path.basename(srcPath)
    const c = classifyFile(srcPath, rel, stat)
    if (c) {
      manifest[c.rel] = c.fp
      if (c.kind === 'text') {
        textFiles.push(c.abs)
      } else {
        otherNames.push(c.rel)
      }
    }
    return { textFiles, otherNames, manifest, truncated }
  }

  const root = srcPath
  let totalSeen = 0

  async function walk(dir, depth) {
    if (truncated || depth > MAX_DEPTH) {
      return
    }
    let dirents
    try {
      dirents = await fs.promises.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const dirent of dirents) {
      if (truncated) {
        return
      }
      const name = dirent.name
      const full = path.join(dir, name)
      if (typeof dirent.isSymbolicLink === 'function' && dirent.isSymbolicLink()) {
        continue
      }
      if (dirent.isDirectory()) {
        if (SKIP_DIRS.has(name) || name.startsWith('.')) {
          continue
        }
        await walk(full, depth + 1)
        continue
      }
      if (!dirent.isFile()) {
        continue
      }
      totalSeen += 1
      if (totalSeen > MAX_FILES) {
        truncated = true
        return
      }
      let stat = null
      try {
        stat = await fs.promises.stat(full)
      } catch {
        /* 仅文件名仍可登记 */
      }
      const c = classifyFile(full, path.relative(root, full), stat)
      if (!c) {
        continue
      }
      manifest[c.rel] = c.fp
      if (c.kind === 'text') {
        textFiles.push(c.abs)
      } else {
        otherNames.push(c.rel)
      }
    }
  }

  await walk(root, 0)
  return { textFiles, otherNames, manifest, truncated }
}

// 把一批文件灌进 KB(文本读正文分批 embed;其余把文件名清单作为一条文本)。复用于首次入库与同步。
async function ingestInto(base, kbDir, name, textFiles, otherNames, sender) {
  const total = textFiles.length
  let done = 0
  for (let i = 0; i < textFiles.length; i += INGEST_BATCH) {
    const slice = textFiles.slice(i, i + INGEST_BATCH)
    const batch = []
    for (const fp of slice) {
      try {
        batch.push({ name: path.basename(fp), buffer: await fs.promises.readFile(fp) })
      } catch {
        /* 读不了的跳过 */
      }
    }
    if (batch.length) {
      await ingestFiles(base, kbDir, batch)
    }
    done += slice.length
    sendProgress(sender, { phase: 'indexing', done, total })
  }
  if (otherNames.length) {
    sendProgress(sender, { phase: 'names', done: total, total })
    const namesText =
      `# ${name} — 未解析正文的文件清单(仅文件名;内容由上游模型按需读取)\n\n` + otherNames.join('\n')
    await ingestFiles(base, kbDir, [{ name: `${kbDir}__filenames.txt`, buffer: Buffer.from(namesText, 'utf-8') }])
  }
}

// 首次入库一个源(文件夹或单文件)。返回 { ok, kb, type, indexed, nameOnly, truncated, manifest }。
async function ingestSource(payload, sender, baseUrl) {
  const base = String(baseUrl || LANGFLOW_URL).replace(/\/+$/, '')
  const srcPath = String((payload && (payload.path || payload.folderPath)) || '').trim()
  const rawName = String((payload && payload.name) || '').trim()

  // 先判文件还是文件夹(都要过 hardening 校验,防逃逸授权根)。
  let resolved
  let isFile = false
  try {
    let st
    try {
      st = await fs.promises.stat(srcPath)
    } catch {
      return { ok: false, error: 'read-error' }
    }
    isFile = st.isFile()
    if (isFile) {
      ;({ resolvedPath: resolved } = await resolveReadableFileForIpc(srcPath, { fs, purpose: 'Knowledge ingest' }))
    } else {
      ;({ resolvedPath: resolved } = await resolveDirectoryForIpc(srcPath, { fs, purpose: 'Knowledge ingest' }))
    }
  } catch (error) {
    return { ok: false, error: error?.code || 'read-error' }
  }

  const name = rawName || path.basename(resolved)
  try {
    sendProgress(sender, { phase: 'preparing', done: 0, total: 0 })
    const kbDir = await ensureKb(base, name)
    const { textFiles, otherNames, manifest, truncated } = await collectFromPath(resolved, isFile)
    await ingestInto(base, kbDir, name, textFiles, otherNames, sender)
    sendProgress(sender, { phase: 'done', done: textFiles.length, total: textFiles.length })
    return {
      ok: true,
      kb: kbDir,
      name,
      // 存**解析后**的路径:manifest 的相对 key 是以它为根算的,同步时也得用同一根才能正确 diff。
      path: resolved,
      type: isFile ? 'file' : 'folder',
      indexed: textFiles.length,
      nameOnly: otherNames.length,
      truncated,
      manifest
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    sendProgress(sender, { phase: 'error', message })
    return { ok: false, error: message }
  }
}

module.exports = {
  ingestSource,
  collectFromPath,
  ingestInto,
  ensureKb,
  deleteKb,
  resolveDirectoryForIpc,
  resolveReadableFileForIpc,
  LANGFLOW_URL
}
