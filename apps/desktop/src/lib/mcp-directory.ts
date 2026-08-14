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
  // GitHub's hosted MCP is intentionally absent. Directory entries are wired
  // through generic Dynamic Client Registration, but GitHub requires each MCP
  // host to provide its own OAuth app (or use a PAT). Advertising it here makes
  // both the composer pill and setup_mcp fail at /register with HTTP 404.
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
  },
  {
    description: 'Deployments, logs, and projects via Vercel’s hosted MCP.',
    docs: 'https://vercel.com/docs/mcp',
    // No `vercel.app` on purpose (same rule as GitHub): pasted deploy-preview
    // links are about the site being previewed, not about managing Vercel.
    hosts: ['vercel.com'],
    keywords: ['vercel'],
    name: 'vercel',
    url: 'https://mcp.vercel.com'
  },
  {
    description: 'Database, auth, and storage from your Supabase projects.',
    docs: 'https://supabase.com/docs/guides/ai-tools/mcp',
    hosts: ['supabase.com', 'supabase.co'],
    keywords: ['supabase'],
    name: 'supabase',
    url: 'https://mcp.supabase.com/mcp'
  },
  {
    description: 'Sites, deploys, and env vars via Netlify’s hosted MCP.',
    docs: 'https://docs.netlify.com/build/build-with-ai/agent-setup-guides/agent-setup-overview/',
    // No `netlify.app` for the same deploy-preview reason as vercel.app.
    hosts: ['netlify.com'],
    keywords: ['netlify'],
    name: 'netlify',
    url: 'https://netlify-mcp.netlify.app/mcp'
  },
  {
    description: 'Models, datasets, Spaces, and papers from the Hugging Face Hub.',
    docs: 'https://huggingface.co/docs/hub/agents-mcp',
    hosts: ['huggingface.co', 'hf.co'],
    keywords: ['hugging face', 'huggingface'],
    // Underscored so prettyName renders "Hugging Face", not "Huggingface".
    name: 'hugging_face',
    url: 'https://huggingface.co/mcp'
  },
  {
    description: 'Tasks, projects, and goals from your Asana workspace.',
    docs: 'https://developers.asana.com/docs/using-asanas-mcp-server',
    hosts: ['asana.com'],
    keywords: ['asana'],
    name: 'asana',
    url: 'https://mcp.asana.com/sse'
  },
  {
    description: 'Conversations, tickets, and customer data from Intercom.',
    docs: 'https://developers.intercom.com/docs/guides/mcp',
    hosts: ['intercom.com', 'intercom.io'],
    keywords: ['intercom'],
    name: 'intercom',
    url: 'https://mcp.intercom.com/mcp'
  },
  {
    description: 'Bases, tables, and records from your Airtable workspace.',
    docs: 'https://support.airtable.com/articles/9897799762-using-the-airtable-mcp-server',
    hosts: ['airtable.com'],
    keywords: ['airtable'],
    name: 'airtable',
    url: 'https://mcp.airtable.com/mcp'
  },
  {
    description: 'Sites, CMS collections, and pages via Webflow’s hosted MCP.',
    docs: 'https://developers.webflow.com/mcp/reference/getting-started',
    // No `webflow.io` — that's published staging sites, not Webflow intent.
    hosts: ['webflow.com'],
    keywords: ['webflow'],
    name: 'webflow',
    url: 'https://mcp.webflow.com/mcp'
  },
  {
    description: 'Payments, invoices, and subscriptions via PayPal’s hosted MCP.',
    docs: 'https://developer.paypal.com/tools/mcp-server/',
    hosts: ['developer.paypal.com'],
    keywords: ['paypal'],
    name: 'paypal',
    url: 'https://mcp.paypal.com/sse'
  },
  {
    description: 'Catalog, orders, and payments via Square’s hosted MCP.',
    docs: 'https://developer.squareup.com/docs/mcp',
    hosts: ['squareup.com'],
    // "square" the English word is everywhere ("square brackets", "square
    // corners") — only the unambiguous brand form triggers.
    keywords: ['squareup'],
    name: 'square',
    url: 'https://mcp.squareup.com/sse'
  }
]

export const directoryEntry = (name: string): McpDirectoryEntry | undefined =>
  MCP_DIRECTORY.find(entry => entry.name === name)
