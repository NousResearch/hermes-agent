/**
 * The desktop's own MCP suggestion directory — deliberately NOT the
 * Nous-approved install catalog (`optional-mcps/`).
 *
 * The catalog is a trust boundary: presence there means a reviewed, pinned
 * manifest, and it only grows via PR. This directory is a different thing —
 * a renderer-local map of well-known OFFICIAL remote MCP endpoints (vendor
 * docs linked per entry) used for two purposes:
 *
 *   1. keyword → suggestion pills over the composer ("you typed jira…"),
 *   2. giving the inline setup card a config to write via the ordinary
 *      `POST /api/mcp/servers` endpoint — the exact same path as pasting the
 *      vendor's snippet into the Capabilities editor by hand.
 *
 * Nothing here changes base Hermes behavior: no backend code reads this file,
 * entries are URL-only remotes (no local process is ever spawned from a
 * suggestion), and every install still lands in config.yaml through the
 * existing validated endpoint. If an entry ALSO exists in the install catalog
 * (e.g. linear, figma), the setup card prefers the catalog path.
 */
export interface McpDirectoryEntry {
  /** Server name as it will appear in mcp_servers config. */
  name: string
  /** Lowercase whole-word/phrase triggers matched against the draft. */
  keywords: string[]
  /** Hostname suffixes that trigger the suggestion when a pasted link points
   *  at the vendor ("yourco.atlassian.net" → atlassian). A pasted URL is the
   *  strongest intent signal there is — stronger than any keyword. */
  hosts?: string[]
  /** Streamable-HTTP/SSE endpoint from the vendor's own docs. */
  url: string
  /** Vendor documentation for the endpoint — shown on the card. */
  docs: string
  description: string
}

export const MCP_DIRECTORY: McpDirectoryEntry[] = [
  {
    description: 'Jira issues and Confluence pages via Atlassian’s hosted MCP.',
    docs: 'https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/',
    hosts: ['atlassian.net', 'atlassian.com', 'jira.com'],
    keywords: ['jira', 'confluence', 'atlassian', 'bitbucket'],
    name: 'atlassian',
    url: 'https://mcp.atlassian.com/v1/sse'
  },
  {
    description: 'Find, create, and update Linear issues and projects.',
    docs: 'https://linear.app/docs/mcp',
    hosts: ['linear.app'],
    keywords: ['linear'],
    name: 'linear',
    url: 'https://mcp.linear.app/mcp'
  },
  {
    description: 'Design context and Code Connect from Figma files.',
    docs: 'https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/',
    hosts: ['figma.com'],
    keywords: ['figma', 'mockup', 'wireframe'],
    name: 'figma',
    url: 'https://mcp.figma.com/mcp'
  },
  {
    description: 'Issues, stack traces, and error context from Sentry.',
    docs: 'https://docs.sentry.io/product/sentry-mcp/',
    hosts: ['sentry.io'],
    keywords: ['sentry', 'stack trace', 'crash report'],
    name: 'sentry',
    url: 'https://mcp.sentry.dev/mcp'
  },
  {
    description: 'Logs, monitors, dashboards, and incidents from Datadog.',
    docs: 'https://docs.datadoghq.com/bits_ai/mcp_server/',
    hosts: ['datadoghq.com', 'datadoghq.eu'],
    keywords: ['datadog', 'apm'],
    name: 'datadog',
    url: 'https://mcp.datadoghq.com/api/unstable/mcp-server/mcp'
  },
  {
    description: 'Repos, issues, and pull requests via GitHub’s hosted MCP.',
    docs: 'https://docs.github.com/en/copilot/customizing-copilot/using-model-context-protocol/using-the-github-mcp-server',
    // No hosts on purpose: github.com links are everywhere in a coding chat
    // (commits, PRs under review, pasted diffs) and would fire constantly.
    keywords: ['github'],
    name: 'github',
    url: 'https://api.githubcopilot.com/mcp/'
  },
  {
    description: 'Pages and databases from your Notion workspace.',
    docs: 'https://developers.notion.com/docs/mcp',
    hosts: ['notion.so', 'notion.site'],
    keywords: ['notion'],
    name: 'notion',
    url: 'https://mcp.notion.com/mcp'
  },
  {
    description: 'Payments, customers, and invoices via Stripe’s hosted MCP.',
    docs: 'https://docs.stripe.com/mcp',
    hosts: ['dashboard.stripe.com'],
    keywords: ['stripe'],
    name: 'stripe',
    url: 'https://mcp.stripe.com'
  }
]

export const directoryEntry = (name: string): McpDirectoryEntry | undefined =>
  MCP_DIRECTORY.find(entry => entry.name === name)
