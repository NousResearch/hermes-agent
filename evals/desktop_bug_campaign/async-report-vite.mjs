import { createServer } from '../../node_modules/vite/dist/node/index.js'
import fs from 'node:fs'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
const root = path.resolve(import.meta.dirname, '../../apps/desktop')
const artifactDir = process.env.ASYNC_REPORT_ARTIFACT_DIR
if (!artifactDir) throw new Error('Set ASYNC_REPORT_ARTIFACT_DIR to an offline artifact directory')
fs.mkdirSync(artifactDir, {recursive:true})
const fixture = path.join(import.meta.dirname, 'async-report-probe.tsx')
const baseline = path.join(artifactDir, 'baseline-hydration.ts')
fs.writeFileSync(baseline, execFileSync('git', ['show', 'a688e7d5ff9aeaaa9c97d28c316467f89ab8c943:apps/desktop/src/lib/chat-messages/hydration.ts'], {encoding:'utf8'}).replaceAll("from './", "from '@/lib/chat-messages/"))
const server = await createServer({
  resolve: {alias: {'@baseline-hydration': baseline}},
  root,
  configFile: path.join(root, 'vite.config.ts'),
  // Keep the node_modules segment: Babel must not compile optimized deps.
  cacheDir: path.join(artifactDir, 'node_modules/.vite'),
  plugins: [{
    name: 'async-report-probe',
    configureServer(server) {
      server.middlewares.use('/async-report-probe.html', async (_req, res, next) => {
        try {
          const source = fs.readFileSync(path.join(import.meta.dirname, 'async-report-probe.html'), 'utf8')
          const html = await server.transformIndexHtml('/async-report-probe.html', source.replace('FIXTURE_ENTRY', `/@fs/${fixture}`))
          res.setHeader('Content-Type', 'text/html')
          res.end(html)
        } catch (error) { next(error) }
      })
    }
  }],
  server: {host: '127.0.0.1', port: 18164, strictPort: true}
})
await server.listen()
server.printUrls()