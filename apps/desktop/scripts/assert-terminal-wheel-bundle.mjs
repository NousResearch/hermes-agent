/** Fail if the packed renderer is missing Page Up/Down or still ships CSI 9001. */
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist')
const assets = path.join(root, 'assets')
const index = readdirSync(assets).find(name => /^index-.*\.js$/.test(name))

if (!index) {
  throw new Error('no renderer index bundle in dist/assets')
}

const text = readFileSync(path.join(assets, index), 'utf8')
const main = readFileSync(path.join(root, 'electron-main.mjs'), 'utf8')

if (!text.includes('[5~') || !text.includes('[6~')) {
  throw new Error(`${index} is missing Page Up/Down CSI`)
}

if (!text.includes('data-tui-scrollbar') || !text.includes('Scroll conversation')) {
  throw new Error(`${index} is missing the conversation scrollbar rail`)
}

if (text.includes('[9001~') || text.includes('\\x1b[9001')) {
  throw new Error(`${index} still encodes CSI 9001`)
}

if (!main.includes('Australia/Sydney') || !main.includes('hermes-term')) {
  throw new Error('electron-main missing Sydney persist attach')
}

if (!main.includes('scroll-up') || !main.includes('copy-mode')) {
  throw new Error('electron-main missing tmux history scroll for Cursor transcripts')
}

console.log(`assert-terminal-wheel-bundle: ${index} Page Up/Down only; persist Sydney; tmux history scroll`)
