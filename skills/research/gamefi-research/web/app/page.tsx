import {
  PR_URL,
  workflowSteps,
  features,
  roadmap,
  sampleProject,
  categoryStyle,
  reportMarkdown,
  stats,
  categoryCounts,
  topProjects,
  scoreBreakdown,
  scoreCeiling,
  relevanceFilter,
  sourceLinks,
  fictionalDisclaimer,
  type Category,
} from "@/lib/sampleData";

const CATEGORY_ORDER: Category[] = ["WATCH", "TEST", "CONTACT", "SKIP"];

function SectionTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="mb-8">
      <p className="text-xs font-semibold uppercase tracking-widest text-sky-400">
        {eyebrow}
      </p>
      <h2 className="mt-2 text-2xl font-bold tracking-tight text-white sm:text-3xl">
        {title}
      </h2>
      {children ? (
        <p className="mt-3 max-w-2xl text-slate-400">{children}</p>
      ) : null}
    </div>
  );
}

export default function Page() {
  return (
    <main className="mx-auto max-w-5xl px-5 py-12 sm:py-16">
      {/* 1. Hero */}
      <section className="rounded-3xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 px-6 py-14 text-center sm:px-12">
        <span className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/60 px-3 py-1 text-xs text-slate-300">
          Hermes Agent · research workflow skill
        </span>
        <h1 className="mx-auto mt-6 max-w-3xl text-4xl font-bold tracking-tight text-white sm:text-5xl">
          GameFi Research Workflow
        </h1>
        <p className="mx-auto mt-3 text-sm font-medium text-slate-500">
          for Hermes Agent
        </p>
        <p className="mx-auto mt-5 max-w-2xl text-lg text-slate-400">
          A structured Hermes workflow for game project discovery, public
          repository review, and clean research summaries.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <a
            href={PR_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-xl bg-sky-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-sky-400"
          >
            View PR
          </a>
          <a
            href="#report"
            className="rounded-xl border border-slate-700 bg-slate-900 px-5 py-2.5 text-sm font-semibold text-slate-200 transition hover:border-slate-500"
          >
            View Sample Report
          </a>
        </div>
        <div className="mt-6 flex flex-wrap justify-center gap-2 text-xs text-slate-500">
          <span className="rounded-full border border-slate-800 px-3 py-1">
            Public repository signals only
          </span>
          <span className="rounded-full border border-slate-800 px-3 py-1">
            Research only
          </span>
          <span className="rounded-full border border-slate-800 px-3 py-1">
            Demo · static sample data
          </span>
        </div>
      </section>

      {/* 2. Problem */}
      <section className="mt-16">
        <SectionTitle eyebrow="The problem" title="Interesting projects get found too late">
          Gaming communities often discover interesting projects too late. This
          workflow helps organize early research using public repository
          signals, documentation quality, project notes, and a clear next-step
          classification.
        </SectionTitle>
      </section>

      {/* 2b. Stats */}
      <section className="mt-16">
        <SectionTitle eyebrow="At a glance" title="Scan summary">
          An illustrative run summary. Counts use the same WATCH / TEST /
          CONTACT / SKIP buckets the scanner produces.
        </SectionTitle>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 px-5 py-6">
            <p className="text-3xl font-bold text-white">{stats.projectsScanned}</p>
            <p className="mt-1 text-sm text-slate-400">Projects scanned</p>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 px-5 py-6">
            <p className="text-3xl font-bold text-white">{stats.topRanked}</p>
            <p className="mt-1 text-sm text-slate-400">Top ranked projects</p>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 px-5 py-6">
            <p className="text-3xl font-bold text-white">{stats.projectsScored}</p>
            <p className="mt-1 text-sm text-slate-400">Projects scored</p>
          </div>
          <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 px-5 py-6">
            <p className="text-lg font-bold text-emerald-300">Public signals only</p>
            <p className="mt-1 text-sm text-slate-400">No private data used</p>
          </div>
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {CATEGORY_ORDER.map((cat) => (
            <div
              key={cat}
              className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3"
            >
              <span
                className={`rounded-md px-2.5 py-1 text-xs font-bold tracking-wide ${categoryStyle[cat]}`}
              >
                {cat}
              </span>
              <span className="text-xl font-bold text-white">
                {categoryCounts[cat]}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* 3. Workflow */}
      <section className="mt-16">
        <SectionTitle eyebrow="How it works" title="The workflow" />
        <div className="flex flex-wrap items-center gap-3">
          {workflowSteps.map((step, i) => (
            <div key={step} className="flex items-center gap-3">
              <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3 text-sm font-medium text-slate-200">
                <span className="mr-2 text-sky-400">{i + 1}</span>
                {step}
              </div>
              {i < workflowSteps.length - 1 ? (
                <span className="text-slate-600">→</span>
              ) : null}
            </div>
          ))}
        </div>
      </section>

      {/* 4. Current Features */}
      <section className="mt-16">
        <SectionTitle eyebrow="Shipped" title="Current features" />
        <div className="grid gap-3 sm:grid-cols-2">
          {features.map((f) => (
            <div
              key={f}
              className="flex items-start gap-3 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3"
            >
              <span className="mt-0.5 text-emerald-400">✓</span>
              <span className="text-sm text-slate-200">{f}</span>
            </div>
          ))}
        </div>
      </section>

      {/* 4b. Top projects table */}
      <section className="mt-16">
        <SectionTitle eyebrow="Ranked" title="Top projects">
          Sorted by AI Research Signal Score. Reasons are short, neutral notes —
          not recommendations.
        </SectionTitle>
        <p className="mb-4 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-amber-200/90">
          {fictionalDisclaimer}
        </p>
        <div className="overflow-x-auto rounded-2xl border border-slate-800">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead>
              <tr className="bg-slate-900/80 text-xs uppercase tracking-wider text-slate-500">
                <th className="px-4 py-3 font-semibold">Project</th>
                <th className="px-4 py-3 font-semibold">Score</th>
                <th className="px-4 py-3 font-semibold">Category</th>
                <th className="px-4 py-3 font-semibold">Reason</th>
              </tr>
            </thead>
            <tbody>
              {topProjects.map((p) => (
                <tr
                  key={p.name}
                  className="border-t border-slate-800 odd:bg-slate-900/30"
                >
                  <td className="px-4 py-3 font-medium text-white">{p.name}</td>
                  <td className="px-4 py-3 font-semibold text-slate-200">
                    {p.score}
                    <span className="text-xs text-slate-500"> / {scoreCeiling}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-md px-2.5 py-1 text-xs font-bold tracking-wide ${categoryStyle[p.category]}`}
                    >
                      {p.category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-400">{p.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 4c. Score breakdown */}
      <section className="mt-16">
        <SectionTitle
          eyebrow="How scoring works"
          title="AI Research Signal Score breakdown"
        >
          The score (0–{scoreCeiling}) reflects research signal strength — how
          much there is to look at and how active/early a project appears — not
          financial merit. It is the sum of these public-signal components.
        </SectionTitle>
        <div className="grid gap-3 sm:grid-cols-2">
          {scoreBreakdown.map((c) => (
            <div
              key={c.label}
              className="rounded-xl border border-slate-800 bg-slate-900/50 p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-white">{c.label}</h3>
                <span className="rounded-md border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-xs font-mono text-sky-300">
                  {c.max}
                </span>
              </div>
              <p className="mt-2 text-sm text-slate-400">{c.detail}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 4d. Relevance filter */}
      <section className="mt-16">
        <SectionTitle eyebrow="Fewer false positives" title="Relevance filter">
          {relevanceFilter.summary}
        </SectionTitle>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
          <p className="text-sm leading-relaxed text-slate-300">
            {relevanceFilter.detail}
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {relevanceFilter.domainTerms.map((t) => (
              <span
                key={t}
                className="rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1 text-xs text-slate-300"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* 5. Sample Output */}
      <section className="mt-16">
        <SectionTitle eyebrow="Example" title="Sample output">
          A single fictional project, shown the way one entry appears after
          scoring.
        </SectionTitle>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-white">
              {sampleProject.name}
            </h3>
            <span
              className={`rounded-md px-2.5 py-1 text-xs font-bold tracking-wide ${categoryStyle[sampleProject.classification]}`}
            >
              {sampleProject.classification}
            </span>
          </div>

          <div className="mt-4">
            <div className="flex items-center gap-3">
              <span className="text-2xl font-bold text-white">
                {sampleProject.score}
              </span>
              <span className="text-sm text-slate-500">/ 100</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-sky-400"
                  style={{ width: `${sampleProject.score}%` }}
                />
              </div>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Research Signal Score — research signal strength, not a
              recommendation.
            </p>
          </div>

          <div className="mt-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Signals
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {sampleProject.signals.map((s) => (
                <span
                  key={s}
                  className="rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1 text-xs text-slate-300"
                >
                  {s}
                </span>
              ))}
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/60 p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Suggested next action
            </p>
            <p className="mt-1 text-sm text-slate-200">
              {sampleProject.nextAction}
            </p>
          </div>
        </div>
      </section>

      {/* 6. Report Preview */}
      <section id="report" className="mt-16 scroll-mt-8">
        <SectionTitle eyebrow="Generated report" title="Report preview">
          The same Markdown format the scanner prototype writes.
        </SectionTitle>
        <pre className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950 p-5 text-xs leading-relaxed text-slate-300">
          {reportMarkdown}
        </pre>
      </section>

      {/* 7. Roadmap */}
      <section className="mt-16">
        <SectionTitle eyebrow="What's next" title="Roadmap" />
        <ul className="grid gap-3 sm:grid-cols-2">
          {roadmap.map((r) => (
            <li
              key={r}
              className="flex items-start gap-3 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3"
            >
              <span className="mt-0.5 text-sky-400">→</span>
              <span className="text-sm text-slate-200">{r}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* 8. Source links */}
      <section className="mt-16">
        <SectionTitle eyebrow="Links" title="Source links">
          Every signal the workflow uses comes from a public source. Sample
          repository links are fictional and included only to show the report
          layout.
        </SectionTitle>
        <div className="grid gap-3 sm:grid-cols-2">
          {sourceLinks.map((l) => (
            <a
              key={l.label}
              href={l.href}
              target="_blank"
              rel="noreferrer"
              className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 transition hover:border-slate-600"
            >
              <p className="text-sm font-medium text-sky-300">{l.label}</p>
              <p className="mt-1 text-xs text-slate-500">{l.note}</p>
            </a>
          ))}
        </div>
      </section>

      <footer className="mt-16 border-t border-slate-800 pt-6 text-xs text-slate-500">
        <p>GameFi Research Workflow for Hermes Agent · demo page · static sample data.</p>
        <p className="mt-1">
          Neutral research showcase. Public repository signals only. All sample
          projects are fictional and for UI demonstration only — every signal is
          automated, unverified, and must be confirmed manually. Not advice of
          any kind.
        </p>
      </footer>
    </main>
  );
}
