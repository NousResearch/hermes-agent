import { createServer } from '../../node_modules/vite/dist/node/index.js'
import fs from 'node:fs'
import path from 'node:path'
const root = path.resolve(import.meta.dirname, '../../apps/desktop')
const artifactDir = process.env.NAVIGATION_ARTIFACT_DIR ?? '/home/teknium/.hermes/cache/desktop-bugs-74848ed3/navigation-markdown'
const fixture = path.join(import.meta.dirname, 'navigation-markdown-probe.tsx')
const server = await createServer({
  root,
  configFile: path.join(root, 'vite.config.ts'),
  // Keep the node_modules segment: Babel must not compile optimized deps.
  cacheDir: path.join(artifactDir, 'node_modules/.vite'),
  plugins: [{
    name: 'navigation-markdown-probe',
    configureServer(server) {
      server.middlewares.use('/navigation-markdown-probe.html', async (_req, res, next) => {
        try {
          const source = fs.readFileSync(path.join(import.meta.dirname, 'navigation-markdown-probe.html'), 'utf8')
          const html = await server.transformIndexHtml('/navigation-markdown-probe.html', source.replace('FIXTURE_ENTRY', `/@fs/${fixture}`))
          res.setHeader('Content-Type', 'text/html')
          res.end(html)
        } catch (error) { next(error) }
      })
    }
  }],
  server: {host: '127.0.0.1', port: 18160, strictPort: true}
})
await server.listen()
server.printUrls()
