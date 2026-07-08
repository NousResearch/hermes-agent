import { useState } from "react";
import { getLogs } from "@/api/endpoints";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

const LEVELS = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"];

export default function LogsPage() {
  const [level, setLevel] = useState("ALL");
  const [lines, setLines] = useState(200);
  const logs = useResource(() => getLogs({ level, lines }), [level, lines]);

  return (
    <ManagementPage
      title="Logs"
      actions={
        <>
          <select
            className="ht-select"
            value={level}
            onChange={(e) => setLevel(e.target.value)}
            aria-label="Log level"
          >
            {LEVELS.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
          <select
            className="ht-select"
            value={lines}
            onChange={(e) => setLines(Number(e.target.value))}
            aria-label="Line count"
          >
            {[100, 200, 500, 1000].map((n) => (
              <option key={n} value={n}>
                {n} lines
              </option>
            ))}
          </select>
          <button type="button" className="ht-btn ht-btn--sm" onClick={logs.reload}>
            Refresh
          </button>
        </>
      }
    >
      <ResourceView resource={logs} empty={(d) => d.lines.length === 0}>
        {(d) => (
          <>
            <p className="ht-muted ht-logs__file">{d.file}</p>
            <pre className="ht-logs">{d.lines.join("\n")}</pre>
          </>
        )}
      </ResourceView>
    </ManagementPage>
  );
}
