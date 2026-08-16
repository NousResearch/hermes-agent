import { describe, expect, it } from 'vitest'

import { parseMcpImport } from './mcp-import'

describe('parseMcpImport', () => {
  describe('mcp.json snippets', () => {
    it('parses an mcpServers-wrapped document', () => {
      const text = JSON.stringify({
        mcpServers: {
          filesystem: { args: ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'], command: 'npx' },
          linear: { url: 'https://mcp.linear.app/sse' }
        }
      })

      expect(parseMcpImport(text)).toEqual([
        {
          config: { args: ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'], command: 'npx' },
          name: 'filesystem'
        },
        { config: { url: 'https://mcp.linear.app/sse' }, name: 'linear' }
      ])
    })

    it('parses a bare name→config map and normalizes the Cursor/Claude type field', () => {
      const text = JSON.stringify({ context7: { type: 'http', url: 'https://mcp.context7.com/mcp' } })

      expect(parseMcpImport(text)).toEqual([
        { config: { transport: 'http', url: 'https://mcp.context7.com/mcp' }, name: 'context7' }
      ])
    })

    it('infers a name for a single unnamed server object', () => {
      const text = JSON.stringify({ args: ['-y', '@modelcontextprotocol/server-github'], command: 'npx' })

      expect(parseMcpImport(text)).toEqual([
        { config: { args: ['-y', '@modelcontextprotocol/server-github'], command: 'npx' }, name: 'github' }
      ])
    })

    it('rejects JSON whose values are not server-shaped', () => {
      expect(parseMcpImport('{"a": 1, "b": "two"}')).toBeNull()
      expect(parseMcpImport('[1, 2, 3]')).toBeNull()
      expect(parseMcpImport('{}')).toBeNull()
    })
  })

  describe('bare command lines', () => {
    it('parses an npx line and names it after the package basename', () => {
      expect(parseMcpImport('npx -y @modelcontextprotocol/server-filesystem /path/to/dir')).toEqual([
        {
          config: { args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir'], command: 'npx' },
          name: 'filesystem'
        }
      ])
    })

    it('strips version pins when inferring the name', () => {
      expect(parseMcpImport('npx -y @upstash/context7-mcp@latest')).toEqual([
        { config: { args: ['-y', '@upstash/context7-mcp@latest'], command: 'npx' }, name: 'context7' }
      ])
    })

    it('parses a uvx line', () => {
      expect(parseMcpImport('uvx mcp-server-fetch')).toEqual([
        { config: { args: ['mcp-server-fetch'], command: 'uvx' }, name: 'fetch' }
      ])
    })

    it('parses a node line and names it after the script', () => {
      expect(parseMcpImport('node ./servers/weather.js --port 3010')).toEqual([
        { config: { args: ['./servers/weather.js', '--port', '3010'], command: 'node' }, name: 'weather' }
      ])
    })

    it('parses a docker run line, keeping placeholder env values verbatim', () => {
      expect(parseMcpImport('docker run -i --rm -e GITHUB_TOKEN=YOUR_KEY mcp/github')).toEqual([
        {
          config: { args: ['run', '-i', '--rm', '-e', 'GITHUB_TOKEN=YOUR_KEY', 'mcp/github'], command: 'docker' },
          name: 'github'
        }
      ])
    })

    it('honors quoted arguments', () => {
      expect(parseMcpImport('npx -y @modelcontextprotocol/server-filesystem "/My Files/docs"')).toEqual([
        {
          config: { args: ['-y', '@modelcontextprotocol/server-filesystem', '/My Files/docs'], command: 'npx' },
          name: 'filesystem'
        }
      ])
    })

    it('parses multiple command lines into multiple entries', () => {
      expect(parseMcpImport('uvx mcp-server-fetch\nnpx -y @modelcontextprotocol/server-memory')).toEqual([
        { config: { args: ['mcp-server-fetch'], command: 'uvx' }, name: 'fetch' },
        { config: { args: ['-y', '@modelcontextprotocol/server-memory'], command: 'npx' }, name: 'memory' }
      ])
    })
  })

  describe('claude mcp add lines', () => {
    it('parses a stdio add with env and a -- separator', () => {
      expect(parseMcpImport('claude mcp add airtable -e AIRTABLE_API_KEY=YOUR_KEY -- npx -y airtable-mcp-server')).toEqual(
        [
          {
            config: {
              args: ['-y', 'airtable-mcp-server'],
              command: 'npx',
              env: { AIRTABLE_API_KEY: 'YOUR_KEY' }
            },
            name: 'airtable'
          }
        ]
      )
    })

    it('parses a stdio add without a -- separator', () => {
      expect(parseMcpImport('claude mcp add linear npx -y mcp-remote https://mcp.linear.app/sse')).toEqual([
        {
          config: { args: ['-y', 'mcp-remote', 'https://mcp.linear.app/sse'], command: 'npx' },
          name: 'linear'
        }
      ])
    })

    it('parses an http transport add pointing at a URL', () => {
      expect(parseMcpImport('claude mcp add --transport http context7 https://mcp.context7.com/mcp')).toEqual([
        { config: { transport: 'http', url: 'https://mcp.context7.com/mcp' }, name: 'context7' }
      ])
    })

    it('parses an sse add with a header', () => {
      expect(
        parseMcpImport('claude mcp add --transport sse api -H "Authorization: Bearer TOKEN_HERE" https://api.example.com/sse')
      ).toEqual([
        {
          config: {
            headers: { Authorization: 'Bearer TOKEN_HERE' },
            transport: 'sse',
            url: 'https://api.example.com/sse'
          },
          name: 'api'
        }
      ])
    })

    it('rejects an add with no command or URL', () => {
      expect(parseMcpImport('claude mcp add lonely')).toBeNull()
    })
  })

  describe('bare URLs', () => {
    it('parses an https URL and names it from the hostname', () => {
      expect(parseMcpImport('https://mcp.linear.app/sse')).toEqual([
        { config: { url: 'https://mcp.linear.app/sse' }, name: 'linear' }
      ])
    })

    it('handles single-label hosts', () => {
      expect(parseMcpImport('http://localhost:8080/mcp')).toEqual([
        { config: { url: 'http://localhost:8080/mcp' }, name: 'localhost' }
      ])
    })
  })

  describe('cursor deeplinks', () => {
    it('decodes the base64 config payload', () => {
      const config = { args: ['-y', '@upstash/context7-mcp'], command: 'npx' }
      const link = `cursor://anysphere.cursor-deeplink/mcp/install?name=context7&config=${btoa(JSON.stringify(config))}`

      expect(parseMcpImport(link)).toEqual([{ config, name: 'context7' }])
    })

    it('normalizes a type field inside the payload', () => {
      const config = { type: 'sse', url: 'https://mcp.example.com/sse' }
      const link = `cursor://anysphere.cursor-deeplink/mcp/install?name=example&config=${btoa(JSON.stringify(config))}`

      expect(parseMcpImport(link)).toEqual([
        { config: { transport: 'sse', url: 'https://mcp.example.com/sse' }, name: 'example' }
      ])
    })

    it('rejects deeplinks with an invalid payload', () => {
      expect(parseMcpImport('cursor://anysphere.cursor-deeplink/mcp/install?name=x&config=!!!notbase64!!!')).toBeNull()
      expect(parseMcpImport('cursor://anysphere.cursor-deeplink/mcp/install?name=x')).toBeNull()
    })
  })

  describe('garbage input', () => {
    it('returns null for prose, empty text, and unknown commands', () => {
      expect(parseMcpImport('')).toBeNull()
      expect(parseMcpImport('   \n  ')).toBeNull()
      expect(parseMcpImport('hello world this is not a config')).toBeNull()
      expect(parseMcpImport('rm -rf /')).toBeNull()
      expect(parseMcpImport('ftp://example.com/thing')).toBeNull()
      expect(parseMcpImport('"unterminated quote npx foo')).toBeNull()
    })
  })
})
