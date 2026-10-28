"use client";

import { useEffect, useId, useState } from 'react';
import { useTheme } from 'fumadocs-ui/provider/base';

export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const { resolvedTheme } = useTheme();
  const [result, setResult] = useState<{ chart: string; theme: string | undefined; svg: string }>();
  useEffect(() => {
    let active = true;
    async function render() {
      const { default: mermaid } = await import('mermaid');
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: resolvedTheme === 'dark' ? 'dark' : 'neutral' });
      const { svg } = await mermaid.render(`mermaid-${id}`, chart);
      if (active) setResult({ chart, theme: resolvedTheme, svg });
    }
    void render().catch((error) => console.error('Mermaid rendering failed', error));
    return () => { active = false; };
  }, [chart, id, resolvedTheme]);
  if (!result || result.chart !== chart || result.theme !== resolvedTheme) return <pre><code>{chart}</code></pre>;
  return <figure className="docs-mermaid" dangerouslySetInnerHTML={{ __html: result.svg }} />;
}
