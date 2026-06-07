import {
  PR_URL,
  workflowSteps,
  features,
  roadmap,
  sampleProject,
  categoryStyle,
  reportMarkdown,
} from "@/lib/sampleData";

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
          Game Research Workflow for Hermes
        </h1>
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

      {/* 5. Sample Output */}
      <section className="mt-16">
        <SectionTitle eyebrow="Example" title="Sample output" />
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

      {/* 8. Links */}
      <section className="mt-16">
        <SectionTitle eyebrow="Links" title="Project links" />
        <a
          href={PR_URL}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-sm text-sky-300 transition hover:border-slate-600"
        >
          Pull Request · NousResearch/hermes-agent #40136
        </a>
      </section>

      <footer className="mt-16 border-t border-slate-800 pt-6 text-xs text-slate-500">
        <p>Game Research Workflow for Hermes · demo page · static sample data.</p>
        <p className="mt-1">
          Neutral research showcase. Public repository signals only. All data
          shown is illustrative and unverified — verify manually.
        </p>
      </footer>
    </main>
  );
}
