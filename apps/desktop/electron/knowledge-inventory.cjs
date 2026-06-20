'use strict'

// 知识库 inventory:递归扫描用户拖入的文件夹,只统计元数据,**不读正文**,给前端出
// "确认范围"报告。入库策略(产品定):
//   - 文本类(langflow embedding 能解析的 TEXT_FILE_TYPES)→ 可全文索引(embed 正文)
//   - 其余任何文件(图片/音视频/Office/二进制…)→ 仅文件名进知识库(不解析正文;真要读
//     内容时由上游多模态模型 GPT/Claude 处理)——所以**不扩 langflow 解析器**
//   - 噪声/系统/临时文件 → 跳过
// 安全 + 护栏:resolveDirectoryForIpc 校验、跳隐藏/依赖目录、跳软链(防逃逸授权根)、
// 文件总数上限,避免扫超大目录卡死。

const fs = require('node:fs')
const path = require('node:path')
const { resolveDirectoryForIpc, resolveReadableFileForIpc } = require('./hardening.cjs')

// 始终跳过的噪声目录(与 fs-read-dir 一致 + 常见缓存/依赖)。
const SKIP_DIRS = new Set([
  '.git',
  '.hg',
  '.svn',
  '.cache',
  '.next',
  '.turbo',
  '.venv',
  '__pycache__',
  'build',
  'dist',
  'node_modules',
  'target',
  'venv'
])

// 与 langflow extract_text_from_bytes 的 TEXT_FILE_TYPES 对齐
// (src/lfx/src/lfx/base/data/utils.py)。这些能 embed 正文;其余文件只记文件名。
const TEXT_EXT = new Set([
  '.csv',
  '.json',
  '.pdf',
  '.txt',
  '.md',
  '.mdx',
  '.yaml',
  '.yml',
  '.xml',
  '.html',
  '.htm',
  '.docx',
  '.py',
  '.sh',
  '.sql',
  '.js',
  '.ts',
  '.tsx'
])

// 纯噪声:连文件名都不值得进知识库 → 跳过。
const NOISE_NAMES = new Set(['.DS_Store', 'Thumbs.db', 'desktop.ini', '.gitkeep'])
const NOISE_EXT = new Set(['.lock', '.tmp', '.temp', '.swp', '.swo', '.part', '.crdownload', '.log'])

const MAX_TEXT_BYTES = 50 * 1024 * 1024 // 文本 >50MB:embed 不划算,降级为"仅文件名"
const MAX_FILES_SCANNED = 100000 // 护栏:扫描文件总数上限,超过即截断
const MAX_DEPTH = 12
const INDEX_FILES_PER_MIN = 120 // 粗略入库速率,仅用于给"预计耗时"一个量级

// 单文件源的盘点报告(支持拖/选单个文件)。
async function scanSingleFile(filePath, fsImpl) {
  let resolved
  try {
    ;({ resolvedPath: resolved } = await resolveReadableFileForIpc(filePath, { fs: fsImpl, purpose: 'Knowledge inventory' }))
  } catch (error) {
    return { ok: false, error: error?.code || 'read-error' }
  }
  const name = path.basename(resolved)
  const ext = path.extname(name).toLowerCase()
  const extLabel = ext.replace(/^\./, '') || '—'
  let size = 0
  try {
    size = (await fsImpl.promises.stat(resolved)).size
  } catch {
    /* 仍可只记文件名 */
  }
  const base = { ok: true, path: resolved, name, truncated: false }
  if (NOISE_NAMES.has(name) || NOISE_EXT.has(ext)) {
    return { ...base, indexable: { count: 0, size: 0, types: [] }, nameOnly: { count: 0, types: [] }, skipped: { hidden: 0, noise: 1 }, estMinutes: 0 }
  }
  if (TEXT_EXT.has(ext) && size <= MAX_TEXT_BYTES) {
    return { ...base, indexable: { count: 1, size, types: [{ ext: extLabel, count: 1, size }] }, nameOnly: { count: 0, types: [] }, skipped: { hidden: 0, noise: 0 }, estMinutes: 1 }
  }
  return { ...base, indexable: { count: 0, size: 0, types: [] }, nameOnly: { count: 1, types: [{ ext: extLabel, count: 1 }] }, skipped: { hidden: 0, noise: 0 }, estMinutes: 0 }
}

async function scanInventory(dirPath, options = {}) {
  const fsImpl = options.fs || fs

  // 先判文件还是文件夹:单文件走单独的盘点;文件夹走递归扫描。
  let rawStat
  try {
    rawStat = await fsImpl.promises.stat(String(dirPath || ''))
  } catch (error) {
    return { ok: false, error: error?.code || 'read-error' }
  }
  if (rawStat.isFile()) {
    return scanSingleFile(dirPath, fsImpl)
  }

  let resolved
  try {
    ;({ resolvedPath: resolved } = await resolveDirectoryForIpc(dirPath, {
      fs: fsImpl,
      purpose: 'Knowledge inventory'
    }))
  } catch (error) {
    return { ok: false, error: error?.code || 'read-error' }
  }

  const textTypes = new Map() // ext -> { count, size }  可全文索引
  const otherTypes = new Map() // ext -> { count }        仅文件名
  let indexableCount = 0
  let indexableSize = 0
  let nameOnlyCount = 0
  let skippedHidden = 0
  let skippedNoise = 0
  let totalSeen = 0
  let truncated = false

  async function walk(dir, depth) {
    if (truncated || depth > MAX_DEPTH) {
      return
    }

    let dirents
    try {
      dirents = await fsImpl.promises.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }

    for (const dirent of dirents) {
      if (truncated) {
        return
      }

      const name = dirent.name
      const full = path.join(dir, name)

      // 软链接一律跳过:防止指向授权根之外、以及环。
      if (typeof dirent.isSymbolicLink === 'function' && dirent.isSymbolicLink()) {
        continue
      }

      if (dirent.isDirectory()) {
        if (SKIP_DIRS.has(name) || name.startsWith('.')) {
          skippedHidden += 1
          continue
        }
        await walk(full, depth + 1)
        continue
      }

      if (!dirent.isFile()) {
        continue
      }

      totalSeen += 1
      if (totalSeen > MAX_FILES_SCANNED) {
        truncated = true
        return
      }

      const ext = path.extname(name).toLowerCase()

      if (NOISE_NAMES.has(name) || NOISE_EXT.has(ext)) {
        skippedNoise += 1
        continue
      }

      let size = 0
      try {
        size = (await fsImpl.promises.stat(full)).size
      } catch {
        // stat 失败不影响"仅文件名"登记。
      }

      if (TEXT_EXT.has(ext) && size <= MAX_TEXT_BYTES) {
        // 文本类且不超大 → 可全文索引。
        indexableCount += 1
        indexableSize += size
        const entry = textTypes.get(ext) || { count: 0, size: 0 }
        entry.count += 1
        entry.size += size
        textTypes.set(ext, entry)
      } else {
        // 非文本(图片/音视频/Office/二进制…)或超大文本 → 仅文件名进库。
        nameOnlyCount += 1
        const key = ext || '—'
        const entry = otherTypes.get(key) || { count: 0 }
        entry.count += 1
        otherTypes.set(key, entry)
      }
    }
  }

  await walk(resolved, 0)

  const textList = [...textTypes.entries()]
    .map(([ext, v]) => ({ ext: ext.replace(/^\./, ''), count: v.count, size: v.size }))
    .sort((a, b) => b.count - a.count)
  const otherList = [...otherTypes.entries()]
    .map(([ext, v]) => ({ ext: ext.replace(/^\./, ''), count: v.count }))
    .sort((a, b) => b.count - a.count)

  return {
    ok: true,
    path: resolved,
    name: path.basename(resolved),
    indexable: { count: indexableCount, size: indexableSize, types: textList },
    nameOnly: { count: nameOnlyCount, types: otherList },
    skipped: { hidden: skippedHidden, noise: skippedNoise },
    estMinutes: indexableCount > 0 ? Math.max(1, Math.ceil(indexableCount / INDEX_FILES_PER_MIN)) : 0,
    truncated
  }
}

module.exports = { scanInventory, TEXT_EXT, SKIP_DIRS, NOISE_NAMES, NOISE_EXT, MAX_TEXT_BYTES }
