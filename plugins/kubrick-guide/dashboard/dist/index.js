(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const REGISTRY = window.__HERMES_PLUGINS__;
  if (!SDK || !REGISTRY) return;

  const React = SDK.React;
  const h = React.createElement;

  const JOBS = [
    { id: "design", title: "Design a cinematic system", detail: "Turn project evidence into a governed visual design specification.", output: "design specification", mode: "design" },
    { id: "script", title: "Develop or diagnose a scene", detail: "Engineer dramatic pressure, observable change, motif behavior, and residue.", output: "scene or script packet", mode: "script" },
    { id: "image", title: "Build an image prompt", detail: "Translate a frame into visible constraints and provider-ready syntax.", output: "image prompt packet", mode: "image" },
    { id: "storyboard", title: "Plan a storyboard or video", detail: "Propagate ownership, geometry, recurrence, and residue across shots.", output: "storyboard and video packets", mode: "storyboard" },
    { id: "qa", title: "Review generated visuals", detail: "Compare expected state to observed frames and propose corrections.", output: "visual QA and correction packet", mode: "visual QA" },
    { id: "continuity", title: "Maintain motif continuity", detail: "Start or update a proposed project ledger without silently changing canon.", output: "motif ledger proposal", mode: "ledger" },
  ];

  const PROVIDERS = [
    ["none", "Neutral / decide later"],
    ["generic", "Generic image model"],
    ["grok-imagine", "Grok Imagine"],
    ["flux", "Flux"],
    ["sd3", "Stable Diffusion 3"],
    ["midjourney", "Midjourney"],
  ];

  function valueOrUnknown(value) {
    const clean = String(value || "").trim();
    return clean || "Not supplied. Ask before inventing.";
  }

  function buildPrompt(state) {
    const job = JOBS.find(function (item) { return item.id === state.job; }) || JOBS[0];
    const lines = [
      "Use the Kubrick skill for this " + job.mode + " task.",
      "",
      "PROJECT",
      "Title: " + valueOrUnknown(state.title),
      "Format: " + valueOrUnknown(state.format),
      "Audience experience: " + valueOrUnknown(state.audience),
      "",
      "DRAMATIC ENGINE",
      "Current pressure or imbalance: " + valueOrUnknown(state.pressure),
      "What must visibly change: " + valueOrUnknown(state.change),
      "What is literally on screen: " + valueOrUnknown(state.visible),
      "Motifs, materials, objects, or recurring forms: " + valueOrUnknown(state.motifs),
      "Constraints and exclusions: " + valueOrUnknown(state.constraints),
      "",
      "PRODUCTION REQUEST",
      "Produce: " + job.output + ".",
      "Scope: " + state.scope + ".",
      "Provider target: " + state.provider + ".",
      "Evidence mode: " + state.evidence + ".",
      "",
      "KUBRICK OPERATING RULES",
      "- Observed form first. Dramatic function first.",
      "- Keep named esoteric or archetypal systems latent unless I explicitly request them.",
      "- Express meaning as enforceable cinematic constraints: geometry, behavior, rhythm, material state, light, sound, convergence, and residue.",
      "- Separate private symbolic architecture from audience-facing prompts.",
      "- Do not invent missing project facts. Ask focused questions or mark the result NOT_COMPUTABLE.",
      "- Treat local ledgers, corrections, and pattern changes as PROPOSED. Never silently rewrite canon.",
      "- Preserve provider-neutral intent before applying provider syntax.",
    ];
    if (state.includeCommands) {
      lines.push("- Include the exact `python scripts/kubrick.py do ...` command when a deterministic CLI run would help.");
    }
    lines.push("", "Start by identifying any missing inputs that block a trustworthy result. Otherwise produce the requested packet and a short next-step checklist.");
    return lines.join("\n");
  }

  function Field(props) {
    const Tag = props.multiline ? "textarea" : "input";
    return h("label", { className: "kg-field" },
      h("span", { className: "kg-label" }, props.label, props.required ? h("b", null, " required") : null),
      props.help ? h("span", { className: "kg-help" }, props.help) : null,
      h(Tag, {
        value: props.value,
        placeholder: props.placeholder || "",
        rows: props.multiline ? (props.rows || 4) : undefined,
        onChange: function (event) { props.onChange(event.target.value); },
      })
    );
  }

  function SelectField(props) {
    return h("label", { className: "kg-field" },
      h("span", { className: "kg-label" }, props.label),
      props.help ? h("span", { className: "kg-help" }, props.help) : null,
      h("select", { value: props.value, onChange: function (event) { props.onChange(event.target.value); } },
        props.options.map(function (option) {
          const pair = Array.isArray(option) ? option : [option, option];
          return h("option", { key: pair[0], value: pair[0] }, pair[1]);
        })
      )
    );
  }

  function navigate(path) {
    window.history.pushState({}, "", path);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }

  function KubrickGuide() {
    const useState = React.useState;
    const [step, setStep] = useState(0);
    const [copied, setCopied] = useState(false);
    const [state, setState] = useState({
      job: "design",
      title: "",
      format: "",
      audience: "",
      pressure: "",
      change: "",
      visible: "",
      motifs: "",
      constraints: "",
      scope: "single frame",
      provider: "none",
      evidence: "Use only supplied and workspace evidence; ask before external research",
      includeCommands: false,
    });

    function update(key, value) {
      setState(function (current) { return Object.assign({}, current, { [key]: value }); });
      setCopied(false);
    }

    const prompt = buildPrompt(state);
    const canContinue = step !== 1 || (state.pressure.trim() && state.change.trim() && state.visible.trim());

    async function copyPrompt(openChat) {
      try {
        await navigator.clipboard.writeText(prompt);
        setCopied(true);
        if (openChat) navigate("/chat");
      } catch (_) {
        setCopied(false);
      }
    }

    return h("div", { className: "kg-shell" },
      h("header", { className: "kg-hero" },
        h("div", { className: "kg-kicker" }, "HERMES CINEMATIC WORKFLOW"),
        h("h1", null, "Build with Kubrick"),
        h("p", null, "Shape a trustworthy cinematic brief, then hand it to Hermes Chat. The wizard keeps symbolic architecture private, visible constraints explicit, and canon changes proposed."),
        h("div", { className: "kg-install" },
          h("span", null, "Need the skill?"),
          h("code", null, "hermes skills install Zero-State-LLC/Kubrick/skills/kubrick"),
          h("button", { onClick: function () { navigator.clipboard.writeText("hermes skills install Zero-State-LLC/Kubrick/skills/kubrick"); } }, "Copy")
        )
      ),
      h("nav", { className: "kg-progress", "aria-label": "Wizard progress" },
        ["Choose job", "Describe change", "Set production", "Review"].map(function (label, index) {
          return h("button", { key: label, className: index === step ? "active" : index < step ? "done" : "", onClick: function () { if (index <= step) setStep(index); } },
            h("span", null, index + 1), label
          );
        })
      ),
      h("main", { className: "kg-panel" },
        step === 0 && h(React.Fragment, null,
          h("div", { className: "kg-heading" }, h("h2", null, "What are you making?"), h("p", null, "Choose the smallest Kubrick surface that matches the outcome.")),
          h("div", { className: "kg-job-grid" }, JOBS.map(function (job) {
            return h("button", { key: job.id, className: "kg-job " + (state.job === job.id ? "selected" : ""), onClick: function () { update("job", job.id); } },
              h("strong", null, job.title), h("span", null, job.detail), h("small", null, "Produces " + job.output)
            );
          }))
        ),
        step === 1 && h(React.Fragment, null,
          h("div", { className: "kg-heading" }, h("h2", null, "Describe observable change"), h("p", null, "Kubrick needs dramatic pressure and visible evidence, not symbol names alone.")),
          h("div", { className: "kg-form two" },
            h(Field, { label: "Project title", value: state.title, onChange: function (v) { update("title", v); }, placeholder: "The Glass Station" }),
            h(Field, { label: "Format", value: state.format, onChange: function (v) { update("format", v); }, placeholder: "Feature film, commercial, storyboard…" }),
            h(Field, { label: "Current pressure or imbalance", required: true, multiline: true, value: state.pressure, onChange: function (v) { update("pressure", v); }, placeholder: "A junior scientist must present evidence while her director controls the room." }),
            h(Field, { label: "What must visibly change", required: true, multiline: true, value: state.change, onChange: function (v) { update("change", v); }, placeholder: "Authority transfers without dialogue; the room begins orienting toward her." }),
            h(Field, { label: "What is literally on screen", required: true, multiline: true, value: state.visible, onChange: function (v) { update("visible", v); }, placeholder: "Conference table, access badge, projection glass, seven observers…" }),
            h(Field, { label: "Audience experience", multiline: true, value: state.audience, onChange: function (v) { update("audience", v); }, placeholder: "Controlled unease resolving into earned recognition." }),
            h(Field, { label: "Motifs and recurring forms", multiline: true, value: state.motifs, onChange: function (v) { update("motifs", v); }, placeholder: "Cracked badge, reflected grid, migrating pool of light…" }),
            h(Field, { label: "Constraints and exclusions", multiline: true, value: state.constraints, onChange: function (v) { update("constraints", v); }, placeholder: "No occult labels, no glowing runes, no generic teal-orange spectacle." })
          ),
          !canContinue && h("p", { className: "kg-warning" }, "Add the pressure, visible change, and on-screen evidence before continuing.")
        ),
        step === 2 && h(React.Fragment, null,
          h("div", { className: "kg-heading" }, h("h2", null, "Set the production contract"), h("p", null, "Kubrick preserves neutral intent before provider adaptation and fails closed when evidence is weak.")),
          h("div", { className: "kg-form" },
            h(SelectField, { label: "Scope", value: state.scope, onChange: function (v) { update("scope", v); }, options: [["single frame", "Single frame"], ["single scene", "Single scene"], ["multi-shot sequence", "Multi-shot sequence"], ["whole project system", "Whole project system"]] }),
            h(SelectField, { label: "Provider", value: state.provider, onChange: function (v) { update("provider", v); }, options: PROVIDERS, help: "Provider selection changes syntax, not the cinematic intent." }),
            h(SelectField, { label: "Evidence mode", value: state.evidence, onChange: function (v) { update("evidence", v); }, options: ["Use only supplied and workspace evidence; ask before external research", "Public research is allowed when it improves the result", "Stay fully offline and use only what I provide"] }),
            h("label", { className: "kg-check" }, h("input", { type: "checkbox", checked: state.includeCommands, onChange: function (event) { update("includeCommands", event.target.checked); } }), h("span", null, h("strong", null, "Include deterministic CLI commands"), " Useful for repeatable compile, ledger, adapter, or QA runs."))
          ),
          h("aside", { className: "kg-rule-card" }, h("strong", null, "What stays governed"), h("ul", null,
            h("li", null, "Audience prompts show observable form, not hidden labels."),
            h("li", null, "Local ledger and learning changes remain PROPOSED."),
            h("li", null, "Missing evidence triggers questions or NOT_COMPUTABLE."),
            h("li", null, "Provider adapters preserve the neutral packet.")))
        ),
        step === 3 && h(React.Fragment, null,
          h("div", { className: "kg-heading" }, h("h2", null, "Review the Hermes prompt"), h("p", null, "Copy it into Chat. Hermes can then load Kubrick and use the appropriate production surface.")),
          h("pre", { className: "kg-prompt" }, prompt),
          h("div", { className: "kg-actions review" },
            h("button", { className: "kg-secondary", onClick: function () { copyPrompt(false); } }, copied ? "Copied" : "Copy prompt"),
            h("button", { className: "kg-primary", onClick: function () { copyPrompt(true); } }, "Copy and open Chat")
          ),
          h("p", { className: "kg-footnote" }, "Clipboard blocked? Select the prompt above and copy it manually. The wizard never sends content outside Hermes on its own.")
        )
      ),
      step < 3 && h("footer", { className: "kg-actions" },
        h("button", { className: "kg-secondary", disabled: step === 0, onClick: function () { setStep(Math.max(0, step - 1)); } }, "Back"),
        h("button", { className: "kg-primary", disabled: !canContinue, onClick: function () { setStep(Math.min(3, step + 1)); } }, step === 2 ? "Review prompt" : "Continue")
      )
    );
  }

  REGISTRY.register("kubrick-guide", KubrickGuide);
})();
