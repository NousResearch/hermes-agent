// Loaded in the builder child before osx-sign. Keep its walk and classifier,
// but own each probe through close: isbinaryfile's path API resolves too early.
// The prebuilder lifecycle also imports this file to validate the supplier pins.
import fs from 'node:fs'
import path from 'node:path'
import { createRequire, registerHooks } from 'node:module'
import { pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const signerEntry = require.resolve('@electron/osx-sign')
const signerRequire = createRequire(signerEntry)
const binaryEntry = signerRequire.resolve('isbinaryfile')
const signerVersion = JSON.parse(fs.readFileSync(path.resolve(path.dirname(signerEntry), '../package.json'), 'utf8')).version
const binaryVersion = JSON.parse(fs.readFileSync(path.resolve(path.dirname(binaryEntry), '../package.json'), 'utf8')).version
if (signerVersion !== '2.4.0' || binaryVersion !== '5.0.7') {
  throw new Error('Revalidate signing probe ownership for the installed osx-sign/isbinaryfile versions')
}

const utilUrl = pathToFileURL(path.join(path.dirname(signerEntry), 'util.js')).href
const needle = `async function getFilePathIfBinary(filePath) {
    if (await isBinaryFile(filePath)) {
        return filePath;
    }
    return null;
}`
const replacement = `// Shared by concurrent walks in this builder child, never by unrelated fs users.
const probeWaiters = [];
let activeProbes = 0;
async function getFilePathIfBinary(filePath) {
    // Sixteen complete probes leave room for Node and codesign at a 64-fd limit.
    if (activeProbes >= 16) await new Promise(resolve => probeWaiters.push(resolve));
    else activeProbes++;
    try {
        const stat = await fs.promises.stat(filePath);
        if (!stat.isFile()) throw new Error('Path provided was not a file!');
        const file = await fs.promises.open(filePath, 'r');
        try {
            // Match isbinaryfile 5.0.7's sample, including its UTF-8 boundary reserve.
            const buffer = Buffer.alloc(515);
            const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
            return await isBinaryFile(buffer, bytesRead) ? filePath : null;
        } finally {
            await file.close();
        }
    } finally {
        const next = probeWaiters.shift();
        if (next) next();
        else activeProbes--;
    }
}`

// Transform just the private supplier helper in memory; never mutate node_modules
// or fs.open. An upstream shape change must fail instead of silently losing the cap.
registerHooks({
  load(url, context, nextLoad) {
    const loaded = nextLoad(url, context)
    if (url !== utilUrl) return loaded
    const source = loaded.source.toString()
    if (!source.includes(needle)) throw new Error('osx-sign binary probe shape changed; revalidate the signing adapter')
    return { ...loaded, source: source.replace(needle, replacement) }
  }
})