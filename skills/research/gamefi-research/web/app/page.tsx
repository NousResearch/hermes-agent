import {
  projects,
  counts,
  meta,
  sampleMarkdown,
  type Category,
} from "@/lib/sampleData";

const CATEGORY_DESC: Record<Category, string> = {
  WATCH: "Interesting signals, needs more observation",
  TEST: "A concrete way to test or review exists",
  CONTACT: "Relevant for creator / community outreach",
  SKIP: "Weak signals, unclear purpose, or off-topic",
};

function Badge({ category }: { category: Category }) {
  return <span className={`badge badge-${category}`}>{category}</span>;
}

function ScoreBar({ score }: { score: number }) {
  return (
    <div className="score">
      <span className="score-num">{score}</span>
      <span className="score-max">/100</span>
      <span className="score-bar">
        <span className="score-fill" style={{ width: `${score}%` }} />
      </span>
    </div>
  );
}

export default function Page() {
  return (
    <main className="container">
      <header className="hero">
        <h1>GameFi Research Workflow</h1>
        <p className="tagline">
          Neutral, structured research on early-stage Web3 game projects — from
          public repository signals to a comparable score and a clear next step.
        </p>
        <div className="pills">
          <span className="pill">Public signals only</span>
          <span className="pill">Research only — not financial advice</span>
          <span className="pill">Demo: static sample data</span>
        </div>
      </header>

      <section className="card">
        <h2>What the scanner does</h2>
        <p>
          The scanner searches public GitHub repositories for early-stage game
          projects, fetches each README, and turns public signals — repository
          activity, freshness, documentation quality, testing/demo indicators,
          and project clarity — into a 0–100 <strong>Game Research Signal
          Score</strong> (a transparent heuristic, not an AI model). A
          lightweight Web3/GameFi relevance filter removes off-topic matches. It
          then classifies each project and produces a Markdown report.
        </p>
      </section>

      <section className="summary">
        <div className="stat">
          <span className="stat-num">{meta.totalScanned}</span>
          <span className="stat-label">scanned (unique)</span>
        </div>
        <div className="stat">
          <span className="stat-num">{counts.TEST}</span>
          <span className="stat-label">TEST</span>
        </div>
        <div className="stat">
          <span className="stat-num">{counts.CONTACT}</span>
          <span className="stat-label">CONTACT</span>
        </div>
        <div className="stat">
          <span className="stat-num">{counts.WATCH}</span>
          <span className="stat-label">WATCH</span>
        </div>
        <div className="stat">
          <span className="stat-num">{counts.SKIP}</span>
          <span className="stat-label">SKIP</span>
        </div>
        <div className="stat">
          <span className="stat-num">{counts.penalized}</span>
          <span className="stat-label">relevance-penalized</span>
        </div>
      </section>

      <section className="card">
        <h2>Category legend</h2>
        <ul className="legend">
          {(Object.keys(CATEGORY_DESC) as Category[]).map((c) => (
            <li key={c}>
              <Badge category={c} /> <span>{CATEGORY_DESC[c]}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h2>Top projects</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Project</th>
                <th>Score</th>
                <th>Category</th>
                <th>Top signals</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => (
                <tr key={p.name}>
                  <td>{p.rank}</td>
                  <td>
                    <a href={p.url} target="_blank" rel="noreferrer">
                      {p.name}
                    </a>
                  </td>
                  <td>
                    <ScoreBar score={p.score} />
                  </td>
                  <td>
                    <Badge category={p.category} />
                  </td>
                  <td className="muted">{p.signals.slice(0, 3).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Project details &amp; score breakdown</h2>
        {projects.map((p) => (
          <div key={p.name} className="project">
            <div className="project-head">
              <a href={p.url} target="_blank" rel="noreferrer">
                {p.name}
              </a>
              <Badge category={p.category} />
              <span className="project-score">{p.score}/100</span>
            </div>
            <p className="reason">{p.reason}</p>
            <p className="muted small">{p.stats}</p>

            <div className="kv">
              <span className="kv-key">Detected signals</span>
              <span className="kv-val">
                {p.signals.length ? p.signals.join(", ") : "none"}
              </span>
            </div>
            <div className="kv">
              <span className="kv-key">Risk notes</span>
              <span className="kv-val">
                {p.risks.length ? p.risks.join(", ") : "none"}
              </span>
            </div>
            <div className="kv">
              <span className="kv-key">Score breakdown</span>
              <span className="kv-val mono">{p.breakdown.join(" · ")}</span>
            </div>
            <div className="kv">
              <span className="kv-key">Sources</span>
              <span className="kv-val">
                <a href={p.sources.repository} target="_blank" rel="noreferrer">
                  repository
                </a>
                {" · "}
                {p.sources.readme.startsWith("http") ? (
                  <a href={p.sources.readme} target="_blank" rel="noreferrer">
                    README
                  </a>
                ) : (
                  <span className="muted">README: {p.sources.readme}</span>
                )}
              </span>
            </div>
          </div>
        ))}
      </section>

      <section className="card">
        <h2>Sample Markdown report</h2>
        <p className="muted small">
          The same format <code>gamefi_scan.py --report</code> writes (excerpt).
        </p>
        <pre className="markdown">{sampleMarkdown}</pre>
      </section>

      <section className="card disclaimer">
        <h2>Manual verification &amp; disclaimer</h2>
        <p>
          All signals are <strong>automated and unverified</strong>. Before
          acting on any entry, open the repository, read the README, and confirm
          the detected signals (testnet / demo / early access, etc.) are real and
          current.
        </p>
        <p className="muted small">
          The Game Research Signal Score reflects research signal strength, not
          financial merit. Neutral research summary — not financial advice, not a
          trading tool, not an investment recommendation. Public signals only.
        </p>
      </section>

      <footer className="footer">
        <span>gamefi-research · Hermes Agent skill</span>
        <span className="muted">Demo UI · static sample data · no secrets</span>
      </footer>
    </main>
  );
}
