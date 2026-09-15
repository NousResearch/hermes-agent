"""Parts 1-12 of the playbook: product, capability, market, offer, ROI, pricing,
acquisition, marketing, message, website."""

from __future__ import annotations

from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, NextPageTemplate, PageBreak, Paragraph,
                                Spacer)

from layout import bullets, caption, contents_flowable, h2, h3, img, section
from theme import (ACCENT, AQUA, CONTENT_W, CRITICAL, GOOD, LADDER, PANEL_HI, S, TEXT_MUTED,
                   WARM, WARNING, Checklist, Flow, Gap, P, Panel, Rule, Timeline, callout,
                   status_chip, table)

W = CONTENT_W


def front_matter():
    # Page 1 uses the "cover" template, which paints everything itself and takes no
    # flowables. NextPageTemplate switches the page AFTER this break, so the cover stays a
    # cover instead of having the executive summary land on top of it.
    s = [NextPageTemplate("body"), Spacer(1, 1), PageBreak()]
    s += [Spacer(1, 4 * mm)]
    s += [Paragraph("EXECUTIVE SUMMARY", S["kicker"]), Paragraph(
        "NOVA, honestly, on one page", S["h1"]), Rule(), Gap(4)]
    s += [P(
        "NOVA is an enterprise AI workforce platform. It deploys governed AI agents into a "
        "company's <b>own</b> AWS account and connects them to that company's knowledge, "
        "systems, communication channels and recurring work — with a policy layer, an "
        "approval queue and an audit log that the customer's own auditor can read. It is "
        "built on the open-source Hermes agent runtime, which NOVA governs rather than "
        "replaces.", "lede")]
    s += [Gap(3)]

    cards = [
        ("PRODUCT", ACCENT,
         "21k lines of NOVA governing a ~600k-line open-source runtime. 13 Control Center "
         "screens, 22 channel platforms, 65 MCP servers, 36 grantable plugins, 60 toolsets, "
         "FTS5 knowledge retrieval with S3 mirroring, and a Terraform module that builds the "
         "whole AWS footprint."),
        ("TARGET MARKET", AQUA,
         "UK accountancy practices with 20–150 staff. 73% of UK firms are turning work away "
         "for lack of people (Advancetrack, 2026); MTD for Income Tax went live on 6 April "
         "2026 and turned compliance into daily work; 95% of mid-tier firms expect more AI "
         "use inside three years (ICAEW, 2026)."),
        ("FIRST OFFER", WARM,
         "“Client Records Chasing & Response” — one painful, measurable workflow, not "
         "“AI agents”. Green / Yellow / Red action classes decided before deployment, so "
         "the partner knows exactly what the workforce may do unsupervised."),
        ("PRICING", ACCENT,
         "Founding customer: £4,500 implementation + £950/month, 12 months, capped scope. "
         "Standard SMB from £7,500 + £1,450/month. Priced on the workflow and the governance, "
         "never on prompts or tokens. Customer pays AWS directly."),
        ("ACQUISITION", AQUA,
         "100 named UK practices, built by hand from ICAEW/ACCA directories, Companies House "
         "and LinkedIn. Multi-touch outreach to the managing partner. 30-day plan with daily "
         "targets in §9. No paid advertising until customer three."),
        ("DEPLOYMENT", WARM,
         "Single EC2 host in the customer's account, no public inbound, SSM-only access, "
         "KMS-encrypted state, IAM with a permissions boundary and per-integration roles "
         "assumed with an ExternalId. Full 36-step guide in §17."),
    ]
    rows = []
    for i in range(0, len(cards), 2):
        pair = []
        for title, colour, body in cards[i:i + 2]:
            pair.append(Panel(
                [Paragraph(f'<font color="#{colour.hexval()[2:]}"><b>{title}</b></font>',
                           S["cell_head"]),
                 Gap(2.5),
                 Paragraph(body, S["cell"])],
                width=(W - 5 * mm) / 2, pad=4.5 * mm, fill=PANEL_HI, accent=colour))
        rows.append(pair)
    s += [table(rows, [(W - 5 * mm) / 2 + 2.5 * mm] * 2, header=False, zebra=False,
                pad=2.4, head_fill=None, body_fill=None)]

    s += [Gap(5), callout(
        "The one thing that is not yet true",
        "NOVA has never been deployed to a real AWS account and no agent in it has ever "
        "called a real model provider. Everything graded below <b>LIVE LOCAL PROVEN</b> was "
        "observed running on a developer machine, in containers, against a scripted "
        "OpenAI-compatible server. That is a real and useful rung — it caught four defects "
        "no unit test could — but it is not a deployment. The first three actions in §24 "
        "exist to close exactly this gap, and until they are closed NOVA should be sold as a "
        "<b>paid pilot</b>, never as a finished product.", "risk")]

    s += [PageBreak(), Paragraph("CONTENTS", S["kicker"]),
          Paragraph("What is in this document", S["h1"]), Rule(), Gap(4),
          contents_flowable()]
    return s


# ---------------------------------------------------------------- PART 1


def part_1():
    s = section("01", "How this document was built",
                "The audit that came before the writing")
    s += [P(
        "Nothing in this playbook was written from memory of what NOVA does. The repository "
        "was read first — source, tests, Terraform, the Docker build, the Control Center, "
        "and eleven internal audit documents — and every capability was placed on a ladder "
        "before a single commercial claim was made. That ladder is the repository's own; it "
        "is defined in <font face='Courier' size='8'>docs/audits/NOVA_HERMES_CAPABILITY_"
        "AUDIT.md</font> and used throughout the codebase.", "lede")]

    s += [h2("The evidence ladder")]
    s += [P(
        "Read these as increasing strength of evidence. The distinction that matters most "
        "commercially is the last one: everything else can be true on a laptop.")]
    rows = [["Rung", "What it means", "What it does NOT mean"]]
    detail = {
        "NOT IMPLEMENTED": ("The capability is absent.", "—"),
        "DECLARED": ("A config key, schema or type exists.", "That anything reads it."),
        "WIRED": ("A runtime call site reads it.", "That the runtime acts on it."),
        "ENFORCED": ("The runtime refuses or acts on it, and bypass is not trivial.",
                     "That it is covered by a test."),
        "TESTED": ("Covered by an automated test in this repository.",
                   "That it has ever run outside a test."),
        "PARTIAL": ("Some of the capability holds; the rest is named and does not.",
                    "That the gap is small."),
        "LIVE LOCAL PROVEN": ("Observed running — real processes, real browser, real files "
                              "— on a developer machine.",
                              "That it works on AWS, or with a real model provider."),
        "LIVE FIELD PROVEN": ("Observed running on a real AWS account with a real model "
                              "provider and real credentials.",
                              "Nothing in NOVA has reached this rung yet."),
    }
    for rung, (means, notmeans) in detail.items():
        rows.append([status_chip(rung), means, notmeans])
    s += [table(rows, [34 * mm, 68 * mm, W - 102 * mm])]
    s += [caption("Source: the ladder is the repository's own, from "
                  "docs/audits/NOVA_HERMES_CAPABILITY_AUDIT.md §Ladder.")]

    s += [h2("What the audit found, in one chart")]
    s += [img("capability-status.png"), caption(
        "Counted from §3 of this document. The right-hand bar is the finding that shapes "
        "every commercial decision here.")]

    s += [h2("The four validations, kept apart")]
    s += [P("These are four different questions and the document never merges them.")]
    rows = [["Validation", "Status", "What was actually done"],
            ["Architecture", status_chip("TESTED"),
             "Boundaries enforced by tests: NOVA depends on stdlib + PyYAML only; runtime "
             "imports confined to nova/runtime/&lt;adapter&gt;/. 19 boundary tests."],
            ["Implementation", status_chip("TESTED"),
             "930 platform tests pass. 6 browser suites (Playwright, real Chromium) pass "
             "against a running control plane, each assertion paired against the files on "
             "disk rather than against the UI's own state."],
            ["Local validation", status_chip("LIVE LOCAL PROVEN"),
             "49 container checks: two tenants, two containers, one image, side by side. "
             "Auth, RBAC, tenant isolation, CSP, graceful shutdown, no secrets in the image."],
            ["AWS validation", status_chip("NOT IMPLEMENTED"),
             "No AWS account has been touched. ECR pull, instance profile, KMS volume "
             "mount, awslogs delivery, SSM access and the ExternalId integration roles are "
             "configured and unexercised."],
            ["Real provider validation", status_chip("NOT IMPLEMENTED"),
             "No agent has called Bedrock or any hosted model. The live worker run used a "
             "scripted OpenAI-compatible server over the runtime's real client."],
            ["Production validation", status_chip("NOT IMPLEMENTED"),
             "No customer, no production load, no 24/7 soak, no backup restore drill on "
             "real infrastructure."]]
    s += [table(rows, [30 * mm, 34 * mm, W - 64 * mm])]
    return s


# ---------------------------------------------------------------- PART 2


def part_2():
    s = section("02", "What NOVA actually is", "In business language, and honestly")

    s += [P(
        "A company buys NOVA to put a small team of AI workers inside its own business — "
        "workers that can read the company's documents, answer from them, send and receive "
        "messages on the company's channels, run work on a schedule, and hand anything "
        "consequential to a human before acting. NOVA is the thing that makes that a "
        "<b>governed</b> arrangement rather than a clever script: who each worker is, what "
        "it may touch, what needs a human, and what it did.", "lede")]

    s += [callout(
        "Say this to customers, and say it early",
        "NOVA did not write the agent runtime. The execution engine underneath — the task "
        "board, the dispatcher, the scheduler, the 22 channel adapters, the plugin system, "
        "MCP — is <b>Hermes</b>, an open-source project of roughly 600,000 lines. NOVA is "
        "about 21,000 lines that sit above it and govern it. This is a strength and should "
        "be sold as one: the customer gets a large, actively developed runtime plus a "
        "governance and control layer that the runtime does not have and was never trying "
        "to be. Claiming to have built all of it is both false and unnecessary.", "warn")]

    s += [h2("The architecture, as a customer should see it")]
    s += [img("architecture.png"), caption(
        "Each layer only talks to the one below it. NOVA's own boundary test suite enforces "
        "the fourth arrow: only nova/runtime/&lt;adapter&gt;/ may import the runtime at all.")]

    s += [h2("How the pieces relate")]
    rows = [["Piece", "Who owns it", "What it does here"],
            ["<b>NOVA</b>", "You", "The tenant bundle (agents, policy, knowledge, channels, "
             "automations as declared YAML), the compiler that turns those into runtime "
             "configuration, the Control Center, RBAC, the audit log."],
            ["<b>Hermes</b>", "NousResearch, open source",
             "Executes. Profiles, durable task board, worker processes, cron store, "
             "channels, plugins, MCP, tool registry. NOVA never forks it: the patch budget "
             "to core is deliberately near zero."],
            ["<b>AWS</b>", "The customer",
             "Where it all runs. One EC2 host, an encrypted EBS volume, KMS, CloudWatch, "
             "IAM, SSM, ECR. Built by NOVA's Terraform module into the customer's account."],
            ["<b>Bedrock</b> (or another provider)", "The customer",
             "The model. NOVA names the model and the credential variable; the customer "
             "supplies the credential. NOVA never stores one."],
            ["<b>Customer systems</b>", "The customer",
             "Reached through channels (22 platforms), MCP servers (65 in the runtime's "
             "curated catalogue) and plugins (36 grantable). Each is a per-agent grant."],
            ["<b>Knowledge</b>", "The customer",
             "Directories of the customer's own documents, optionally mirrored from an S3 "
             "bucket. Indexed into SQLite FTS5 with BM25 ranking — keyword retrieval, not "
             "embeddings. Agents cite what they used."],
            ["<b>Agents</b>", "You configure, customer approves",
             "A NOVA agent <i>is</i> a Hermes profile. That is why per-agent state, "
             "credentials, schedules and memory are isolated by directory rather than by a "
             "check somebody could forget to write."],
            ["<b>Control Center</b>", "You",
             "The customer's only surface. 13 screens. The runtime's own web dashboard is "
             "an operator console with zero concept of a tenant and must never be exposed "
             "to a customer."]]
    s += [table(rows, [38 * mm, 34 * mm, W - 72 * mm])]

    s += [h2("Deployment models")]
    rows = [["Model", "Status", "What it means"],
            ["<b>1. Customer-owned AWS (BYOC)</b>", status_chip("TESTED"),
             "The Terraform module builds into the customer's account. Their data never "
             "leaves it; they hold the KMS key; they pay AWS directly. <b>This is the "
             "model to sell.</b> The module is complete and unit-tested; it has not yet "
             "been applied."],
            ["<b>2. Managed NOVA deployment</b>", status_chip("PARTIAL"),
             "The same single-tenant stack, in an account you operate on the customer's "
             "behalf. Technically identical; commercially it makes you the data processor "
             "and brings a security review, a DPA and an on-call rota you do not yet have."],
            ["<b>3. Multi-tenant managed platform</b>", status_chip("NOT IMPLEMENTED"),
             "One deployment serving many customers. <b>The code does not support this and "
             "the architecture deliberately does not.</b> State is SQLite on one attached "
             "volume; the guarantees hold across processes on one host, not across hosts. "
             "Tenant isolation today is one tenant per deployment. Do not sell it."]]
    s += [table(rows, [40 * mm, 32 * mm, W - 72 * mm])]

    s += [Gap(3), callout(
        "Why single-tenant is the right first answer anyway",
        "The buyer you are targeting is a partner in a professional-services firm who is "
        "personally liable for client confidentiality. “It runs in your own AWS account, "
        "your data never reaches us, and you can revoke our access by deleting one IAM "
        "role” is a far easier sentence to sell than any multi-tenancy story. The "
        "architectural limitation and the commercial advantage are the same fact.", "good")]
    return s


# ---------------------------------------------------------------- PART 3


def part_3():
    s = section("03", "Capability catalogue", "What NOVA can do, graded")

    s += [P(
        "Every capability below was verified against source and tests. Where the Control "
        "Center shows something, the screen is named. Where a capability exists in the "
        "runtime but NOVA does not surface it, that is said rather than implied — a UI "
        "placeholder is not a feature.", "lede")]

    def cap_table(rows):
        head = [["Capability", "Status", "What it does · business value · what production needs"]]
        return table(head + rows, [36 * mm, 30 * mm, W - 66 * mm], font_size=7.6)

    # --- workforce
    s += [h2("3.1  AI workforce")]
    s += [cap_table([
        ["Agents (as runtime profiles)", status_chip("LIVE LOCAL PROVEN"),
         "An agent is declared in YAML and compiled into a runtime profile with its own "
         "directory, config, persona, policy and credentials. <b>Value:</b> a named worker "
         "with a job description the customer can read and change. <b>Production:</b> "
         "model credentials per profile."],
        ["Agent CRUD from the browser", status_chip("LIVE LOCAL PROVEN"),
         "Create, edit, duplicate, archive, restore, delete — all writing to the tenant "
         "bundle, then applying. <b>Value:</b> the customer changes the workforce without "
         "an engineer. <b>Governance:</b> admin-only, every write audited intent→committed."],
        ["Persona / “Soul” editing", status_chip("LIVE LOCAL PROVEN"),
         "The standing instruction each agent carries on every turn, edited through the "
         "real backend and capped at 20,000 characters. Written to the bundle, never to the "
         "derived SOUL.md — which apply overwrites."],
        ["Objectives → task graph", status_chip("TESTED"),
         "A declared objective compiles into ordered steps on the runtime's board. "
         "<b>Value:</b> multi-step work a human can watch. <b>Gap:</b> the full dependency "
         "graph is in the runtime; NOVA surfaces steps, not the graph."],
        ["Durable task board", status_chip("LIVE LOCAL PROVEN"),
         "Tasks and run history survive process and container restarts (runtime's SQLite "
         "Kanban store). Surfaced on the Work screen, tenant-scoped."],
        ["Task detail (runs, comments, artifacts)", status_chip("TESTED"),
         "Per-task history. <b>Security:</b> the artifact's host path is deliberately never "
         "returned by the API."],
        ["Worker processes / isolation", status_chip("LIVE LOCAL PROVEN"),
         "Each agent runs in its own OS process with its own working directory. Proven by "
         "running real dispatched workers, which is how four seam defects were found."],
        ["Supervisor / routing", status_chip("TESTED"),
         "Objectives route to agents by declared capability; delegation is validated "
         "against the roster at load, so a typo fails at configuration time."],
        ["Agent memory", status_chip("WIRED"),
         "The runtime keeps curated per-profile memory files. <b>NOVA surfaces nothing.</b> "
         "Do not sell agent memory."],
        ["Agent status", status_chip("LIVE LOCAL PROVEN"),
         "Enabled/disabled, applied-vs-saved, profile materialised, scheduler liveness. "
         "Only states backed by real runtime data are shown."],
    ])]

    # --- governance
    s += [h2("3.2  Governance")]
    s += [cap_table([
        ["RBAC (viewer / admin)", status_chip("LIVE LOCAL PROVEN"),
         "Two roles; an exact-match route table where anything undeclared requires admin. "
         "<b>Value:</b> staff can watch, partners can change. Proven in-container: viewer "
         "403s on policy, decisions, budget and every write."],
        ["Named principals, hashed tokens", status_chip("LIVE LOCAL PROVEN"),
         "<font face='Courier' size='7'>nova token new</font> mints a token, prints it once "
         "and stores only its SHA-256. <b>NOVA never holds the credential.</b> The audit "
         "log records the human, not a service name."],
        ["Policy compile → decide", status_chip("LIVE LOCAL PROVEN"),
         "A tenant policy compiles per agent into a document the enforcement plugin reads "
         "at every tool call. Fail-closed: a missing or unreadable policy denies, it does "
         "not fall open."],
        ["Tool permissions & deny rules", status_chip("ENFORCED"),
         "Denials compile into the runtime's own unconditional deny list, evaluated before "
         "any bypass mode. <b>Honest limit:</b> positive scoping (toolsets/allow) is "
         "recorded but <b>not</b> enforced by the adapter, and the materialiser says so."],
        ["Approvals queue", status_chip("TESTED"),
         "Actions a deployment requires a human to approve stop and queue. Release and "
         "refusal are admin-only and audited."],
        ["Audit logging", status_chip("LIVE LOCAL PROVEN"),
         "Write-ahead: <i>intent</i> before the act, <i>committed</i> or <i>failed</i> "
         "after, with the authenticated human and a correlation id. Rotation, retention and "
         "tamper-evidence implemented. <b>Value:</b> this is what a partner's PI insurer "
         "asks for."],
        ["Tenant isolation", status_chip("LIVE LOCAL PROVEN"),
         "One tenant per deployment; per-agent state isolated by directory. Proven with two "
         "containers side by side: each token 401s against the other tenant, automations "
         "and provenance are invisible across the boundary."],
        ["Automation governance", status_chip("TESTED"),
         "An automation is a <i>spec</i> that compiles — its permissions, knowledge and "
         "channels must be a subset of the owning agent's. There is no path from the "
         "Control Center to a raw scheduled prompt."],
        ["Per-channel approval policy", status_chip("TESTED"),
         "A channel can require approval for actions that would otherwise pass — e.g. "
         "anything that sends externally."],
        ["Capability grants (MCP / plugins)", status_chip("LIVE LOCAL PROVEN"),
         "Per-agent grants declared in the bundle and compiled by apply, so they survive "
         "re-application. Admin-only, audited."],
    ])]

    # --- knowledge
    s += [h2("3.3  Knowledge (RAG)")]
    s += [cap_table([
        ["Corpora & ingestion", status_chip("LIVE LOCAL PROVEN"),
         "A corpus is a directory plus include/exclude globs, a size cap and a "
         "classification label. The walk refuses symlinks, anything resolving outside the "
         "root, and dot-directories — this is the security boundary of the capability."],
        ["Retrieval architecture", status_chip("TESTED"),
         "SQLite <b>FTS5</b>, porter stemmer, <b>BM25</b> ranking. <b>No embeddings, "
         "deliberately.</b> <b>Sell it as:</b> keyword search that cites its source. "
         "<b>Do not sell:</b> semantic/vector search. Say so before a technical buyer asks."],
        ["Browser upload", status_chip("LIVE LOCAL PROVEN"),
         "Documents uploaded through the Control Center, name rebuilt not trusted "
         "(traversal and Unicode-mangling both tested), re-indexed on upload so “stored” "
         "never silently means “invisible”."],
        ["S3-mirrored corpora", status_chip("TESTED"),
         "A corpus declares an S3 origin; NOVA mirrors it and the existing ingest runs over "
         "the result. Keys escaping the root, unaccepted types and oversized objects are "
         "refused and named. <b>Proven against a real boto3 client</b> over an in-process "
         "S3 server — not on real S3."],
        ["Corpus permissions", status_chip("ENFORCED"),
         "An agent reads only corpora its spec grants, validated at bundle load. Declaring "
         "a corpus also grants the search tool — a trap found by running a live worker."],
        ["Citations", status_chip("TESTED"),
         "Retrieved chunks carry their document, so an answer can name where it came from. "
         "<b>Value:</b> in a professional-services firm an uncited answer is unusable."],
        ["Document types", status_chip("PARTIAL"),
         "Text and Markdown are first-class. Extraction of richer formats is a runtime "
         "capability NOVA reports via a capability flag; it has not been exercised end to "
         "end. <b>Scope PDFs explicitly in a pilot, do not assume them.</b>"],
    ])]

    # --- automation
    s += [h2("3.4  Automation")]
    s += [cap_table([
        ["Governed automations", status_chip("TESTED"),
         "Declared, compiled, then created in the runtime's own cron store under the owning "
         "agent's profile. The objective is capped at 4,000 characters."],
        ["Pause / resume / delete", status_chip("LIVE LOCAL PROVEN"),
         "Admin-only, audited, surfaced per agent and on the Automations screen."],
        ["Automation provenance", status_chip("LIVE LOCAL PROVEN"),
         "A tenant-salted digest registry records what NOVA created and why. Survives a "
         "container restart — verified."],
        ["Execution ledger", status_chip("WIRED"),
         "The runtime keeps per-execution rows. NOVA reads the agent's own ledger directly "
         "(the runtime's own accessor resolves to the wrong store — a real bug found and "
         "worked around). <b>Surfaced against seeded data only.</b>"],
        ["<b>Schedules actually firing</b>", status_chip("NOT IMPLEMENTED"),
         "<b>Read this twice.</b> The cron ticker lives inside the runtime's gateway; there "
         "is no standalone daemon. A deployment can hold a perfectly correct schedule that "
         "<b>nothing executes</b>. The Automations screen leads with scheduler liveness and "
         "says “nothing is running these schedules” when nothing is. <b>Running a gateway "
         "process is a deployment step that has never been done, and it blocks any pilot "
         "that promises scheduled work.</b>"],
    ])]

    # --- communication
    s += [h2("3.5  Communication channels")]
    s += [cap_table([
        ["Provider catalogue", status_chip("LIVE LOCAL PROVEN"),
         "22 platforms, read from the runtime's own plugin manifests rather than a "
         "hand-kept list. Each carries how its support was established."],
        ["<b>Verification status</b>", status_chip("DECLARED"),
         "<b>Zero of the 22 are field-validated.</b> 5 were read in source (Slack, Discord, "
         "Telegram, Email, WhatsApp); 17 are manifest-only. A test asserts that no provider "
         "may claim field validation, so promoting one requires evidence. <b>Never put a "
         "channel logo on a slide without connecting it first.</b>"],
        ["Grant enforced twice", status_chip("ENFORCED"),
         "NOVA's parser refuses a route outside the allowed agents, and the grant compiles "
         "into the runtime's own fail-closed allowlist. A validation you can bypass by "
         "editing YAML is advice; a runtime that will not deliver is a control."],
        ["Credential isolation", status_chip("PARTIAL"),
         "<b>Stated precisely, because the nuance matters:</b> credentials in a per-agent "
         "store are isolated. A credential exported into the <i>host</i> environment is "
         "readable by every agent on that host — verified. Isolation is a deployment "
         "property, and the readiness report says where each credential resolved from."],
        ["No secret can enter a declaration", status_chip("ENFORCED"),
         "token, api_key, secret, password, credential and app_secret are refused at parse "
         "time at any nesting depth, and stripped from what NOVA writes."],
        ["Webhook transport", status_chip("DECLARED"),
         "WhatsApp and similar need a publicly reachable HTTPS endpoint. The AWS module has "
         "<b>no inbound rules at all</b>. That is a deliberate deployment decision nobody "
         "has made yet."],
    ])]

    # --- mcp / plugins
    s += [h2("3.6  MCP and plugins")]
    s += [cap_table([
        ["MCP catalogue", status_chip("LIVE LOCAL PROVEN"),
         "65 curated servers from the runtime's pinned manifests. Granting is per agent, "
         "declared in the bundle, compiled by apply. Tools arrive as a toolset named "
         "<font face='Courier' size='7'>mcp-&lt;server&gt;</font> and are decided by the "
         "same policy as any other tool."],
        ["<b>OAuth reality</b>", status_chip("PARTIAL"),
         "<b>54 of the 65 authenticate with OAuth</b>, which needs a browser consent the "
         "Control Center cannot perform. A granted OAuth server is reported as <i>configured, "
         "not yet authorized</i> with the exact command that completes it. 10 need no auth "
         "and work immediately."],
        ["Catalogue-only security model", status_chip("ENFORCED"),
         "Arbitrary MCP servers cannot be added. A stdio server is a command line; accepting "
         "one from a browser form would be remote code execution with a form around it. The "
         "one catalogue entry that clones and builds is refused too."],
        ["Plugin grants", status_chip("LIVE LOCAL PROVEN"),
         "36 grantable of 58 discovered. Three states, because the runtime has three: "
         "bundled backends load unless disabled, so “not listed” and “disabled” are "
         "different answers. Channels are excluded — they have their own screen."],
        ["Plugin enable merge safety", status_chip("TESTED"),
         "A plugin grant merges into the enabled list rather than replacing it. Replacing it "
         "would silently drop NOVA's own policy plugin — the most dangerous state a "
         "governance control can be in, because it passes review by inspection. This "
         "actually happened once and is now pinned by a test."],
    ])]

    # --- aws
    s += [h2("3.7  AWS deployment")]
    s += [cap_table([
        ["Terraform module", status_chip("TESTED"),
         "13 resources: EC2, EBS, KMS key + alias, CloudWatch log group, security group "
         "(egress 443 only, <b>no ingress</b>), 2 IAM roles, 2 role policies, a permissions "
         "boundary policy, an attachment and an instance profile. 16 input variables."],
        ["Docker image", status_chip("LIVE LOCAL PROVEN"),
         "304 MB, non-root uid 10001, no apt layer, health check in the standard library, "
         "immutable tagging, state on a declared volume. 49 checks pass including 8 for "
         "“no secrets in the image”."],
        ["ECR", status_chip("DECLARED"),
         "Not created by the module; the instance role is granted pull. <b>Nothing has been "
         "pushed to any registry.</b>"],
        ["IAM design", status_chip("TESTED"),
         "A runtime role with <b>no customer-data permissions</b>, under an explicit "
         "permissions boundary so a future mistaken attachment cannot widen it. One role per "
         "integration, each assumed with <font face='Courier' size='7'>sts:ExternalId</font>."],
        ["SSM Session Manager", status_chip("DECLARED"),
         "The only way in. No inbound port, no SSH key, no bastion. Configured, never used."],
        ["IMDSv2 required", status_chip("TESTED"),
         "Asserted by the module's tests. Closes the classic SSRF-to-credentials path."],
        ["KMS encryption", status_chip("TESTED"),
         "Volume and logs. The customer may supply their own key ARN — and for a "
         "professional-services buyer, they should."],
        ["Bedrock", status_chip("DECLARED"),
         "Model ids enumerated, never wildcarded. <b>No model has ever been invoked.</b>"],
        ["Secrets Manager", status_chip("DECLARED"),
         "Readable under one prefix and nowhere else."],
    ])]

    # --- control center
    s += [h2("3.8  Control Center")]
    s += [P("13 screens. Everything a customer sees comes through NOVA's own tenant-scoped "
            "control API — never the runtime's operator dashboard, which has no concept of "
            "a tenant at all.")]
    s += [cap_table([
        ["Overview · Agents · Work · Objectives", status_chip("LIVE LOCAL PROVEN"),
         "The daily surface. Agent profile pages carry Overview, Work, Soul, Model, "
         "Capabilities, Schedules, Activity, Knowledge, Channels, Extensions, Permissions "
         "and Usage."],
        ["Approvals · Activity", status_chip("TESTED"),
         "The queue, and every refusal and escalation the policy layer recorded. Permitted "
         "calls are deliberately not logged as decisions."],
        ["Knowledge · Channels · Automations", status_chip("LIVE LOCAL PROVEN"),
         "Corpora with upload, sync and reindex; the channel catalogue with per-agent "
         "grants; schedules with liveness stated first."],
        ["Policies · Usage", status_chip("TESTED"),
         "Admin-only. Usage reports observed token and estimated cost."],
        ["<b>Usage is not a spending cap</b>", status_chip("PARTIAL"),
         "<b>Never sell it as one.</b> Run-budget seconds trigger a soft wrap-up message; "
         "nothing terminates. Token and cost figures are observed, not enforced. A hard "
         "spending ceiling is structurally unavailable and the repository says so."],
        ["Settings & branding", status_chip("LIVE LOCAL PROVEN"),
         "Company details, brand colour, logo — applied across the dashboard and into every "
         "agent's persona, so agents introduce themselves as the customer's product. "
         "White-labelling is real and works."],
        ["Security posture", status_chip("LIVE LOCAL PROVEN"),
         "Strict CSP (no inline script, no external anything), X-Frame-Options DENY, bearer "
         "auth, cross-origin writes refused, form-encoded writes refused."],
    ])]
    return s


# ---------------------------------------------------------------- PART 4


def part_4():
    s = section("04", "The best first UK market", "Researched, scored, chosen")

    s += [P(
        "The choice below is made on evidence about the UK market in 2026, not on which "
        "sector sounds most exciting. Sources are cited inline and listed in §25. Where a "
        "figure is an estimate rather than a published statistic, it says so.", "lede")]

    s += [h2("4.1  What the UK market actually looks like")]
    rows = [["Finding", "Figure", "Source", "Type"],
            ["UK businesses (10+ employees) using at least one AI technology",
             "~35%, up from ~12% in Sept 2023", "ONS, <i>Artificial intelligence in UK "
             "businesses</i>, 20 July 2026", "Published"],
            ["All size bands", "29%", "ONS, same release", "Published"],
            ["Adoption is <b>shallow</b>: average AI technologies per adopting business",
             "~1.4 → ~1.6 since late 2023", "ONS, same release", "Published"],
            ["UK SMEs using AI in some form / using it extensively to automate operations",
             "54% / <b>11%</b>", "British Chambers of Commerce, March 2026", "Published"],
            ["AI-using businesses reporting increased revenue", "12%",
             "ONS / DSIT, 2026", "Published"],
            ["UK SMEs (0–249 employees)", "~5.68 million",
             "UK Business Population Estimates 2025, via House of Commons Library", "Published"],
            ["UK accounting &amp; auditing businesses", "~29,600 (IBISWorld) to ~35,900 "
             "(Companies House–derived)", "IBISWorld 2025; Firmbase 2026", "Published, two definitions"],
            ["UK accounting firms turning away clients for lack of staff",
             "<b>73%</b>", "Advancetrack <i>Accounting Talent Index</i> 2026", "Published"],
            ["Firms saying sustained workloads could push people out of the profession",
             "74%", "Advancetrack, same index", "Published"],
            ["Mid-tier firms expecting increased AI use within 3 years", "<b>95%</b>",
             "ICAEW mid-tier firms research, 35 firms, Feb–Mar 2026", "Published"],
            ["Firms whose strategy explicitly includes AI adoption", "86%",
             "ICAEW, same research", "Published"],
            ["Firms that believe they can assess AI's workforce impact", "17%",
             "ICAEW, same research", "Published"],
            ["MTD for Income Tax live for sole traders/landlords above £50,000",
             "from <b>6 April 2026</b>", "HMRC Making Tax Digital timetable", "Published"]]
    s += [table(rows, [58 * mm, 30 * mm, W - 118 * mm, 20 * mm], font_size=7.4)]

    s += [Gap(3), callout(
        "The single most useful number in this table",
        "<b>54% of UK SMEs use AI in some form; 11% use it extensively to automate "
        "operations</b> (BCC, March 2026), and adoption is shallow — about 1.6 technologies "
        "per adopting business (ONS, July 2026). The market has bought chat assistants and "
        "stopped. Almost nobody has deployed a governed workforce that touches real systems. "
        "That is precisely the gap NOVA is built for, and it means the sales conversation is "
        "<i>not</i> “have you considered AI” — it is “you tried AI, it saved drafting time, "
        "and the admin came straight back. Here is why.”", "good")]

    s += [h2("4.2  Scoring the candidate verticals")]
    s += [P(
        "Twelve factors, each scored 1–10, weighted by how much they decide a <i>first</i> "
        "customer rather than a tenth. Weights are my judgement and are stated so you can "
        "disagree with them; the underlying observations are sourced above and in §25.")]

    rows = [["Factor", "Weight", "Why it is weighted that way"],
            ["Repetitive, documentable workflows", "1.5×", "Decides whether NOVA can do the job at all."],
            ["ROI measurability", "1.5×", "You need a case study more than you need revenue."],
            ["Accessible decision maker", "1.4×", "Owner/partner who can sign without a committee."],
            ["Willingness to pay", "1.3×", "Firm must already buy software and outsourced labour."],
            ["Document volume", "1.2×", "Feeds the knowledge capability, which is real today."],
            ["Communication volume", "1.2×", "Where the hours actually go."],
            ["Implementation time", "1.1×", "A pilot that takes four months kills momentum."],
            ["Case-study likelihood", "1.1×", "Will they let you name them?"],
            ["AI adoption / readiness", "1.0×", "Warm enough to answer, not so warm it is crowded."],
            ["Technical readiness", "0.9×", "Existing CRM/practice software to integrate with."],
            ["Integration complexity (inverted)", "0.9×", "Fewer bespoke connectors to build."],
            ["Regulatory risk (inverted)", "1.2×", "A regulated buyer adds months of review."]]
    s += [table(rows, [58 * mm, 16 * mm, W - 74 * mm], font_size=7.6)]

    s += [Gap(4), img("market-scores.png"), caption(
        "Weighted composite of the twelve factors above. Scores are my assessment against "
        "the evidence in §4.1 and §25 — they are a decision tool, not a measurement.")]

    s += [PageBreak(), h2("4.3  The scoring detail")]
    verticals = [
        ("Accountancy practices", [9, 9, 9, 8, 9, 9, 8, 9, 8, 8, 7, 6], "8.4"),
        ("Property / block management", [9, 8, 8, 7, 7, 9, 7, 7, 6, 6, 7, 7], "7.5"),
        ("Recruitment agencies", [9, 8, 8, 7, 6, 9, 8, 7, 7, 7, 7, 6], "7.3"),
        ("IT managed service providers", [8, 8, 8, 7, 6, 8, 7, 6, 8, 9, 8, 6], "7.1"),
        ("Digital / marketing agencies", [8, 7, 9, 6, 6, 8, 8, 7, 9, 8, 7, 5], "6.8"),
        ("Logistics &amp; freight", [8, 8, 6, 7, 7, 8, 6, 6, 5, 6, 6, 6], "6.5"),
        ("Professional services (other)", [7, 7, 7, 7, 8, 7, 7, 6, 6, 6, 6, 6], "6.4"),
        ("E-commerce operations", [8, 8, 7, 5, 4, 9, 7, 6, 8, 8, 6, 5], "6.0"),
        ("Legal firms", [8, 6, 5, 8, 9, 7, 4, 5, 5, 5, 5, 3], "5.2"),
        ("Financial services (FCA)", [8, 6, 3, 9, 9, 7, 3, 4, 6, 7, 4, 1], "4.4"),
    ]
    head = ["Vertical", "Wkflw", "ROI", "DM", "Pay", "Docs", "Comms", "Impl", "Case",
            "Adopt", "Tech", "Integ", "Reg", "Score"]
    rows = [head]
    for name, scores, total in verticals:
        rows.append([name] + [str(x) for x in scores] + [f"<b>{total}</b>"])
    widths = [38 * mm] + [(W - 38 * mm - 14 * mm) / 12] * 12 + [14 * mm]
    s += [table(rows, widths, font_size=6.6, pad=2.6,
                align={i: "CENTER" for i in range(1, 14)})]
    s += [caption("Columns in the order of the weight table in §4.2. Regulatory and "
                  "integration columns are already inverted (10 = low risk / low complexity).")]

    s += [h2("4.4  The decision")]

    s += [Panel([
        Paragraph('<font color="#%s"><b>#1 TARGET — UK ACCOUNTANCY PRACTICES, 20–150 STAFF</b></font>'
                  % GOOD.hexval()[2:], S["h3"]),
        P("Independent and mid-tier practices outside the Top 50. Not the Big Four, not "
          "one-person bookkeepers."),
        P("<b>Why it wins on evidence, not vibes:</b>"),
        Paragraph("<bullet>&#8226;</bullet><b>The pain is acute and documented.</b> 73% of "
                  "UK firms are turning away clients for lack of staff and 74% think "
                  "workloads could push people out of the profession (Advancetrack 2026). "
                  "That is not a nice-to-have budget, it is a capacity crisis.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>The timing is externally forced.</b> MTD for "
                  "Income Tax went live on 6 April 2026 and turned annual compliance into "
                  "daily work. The chasing volume went up this year and is not going back "
                  "down.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>The buyer is already convinced of the "
                  "category.</b> 95% of mid-tier firms expect more AI, 86% have it in "
                  "strategy — but only 17% believe they can assess its impact (ICAEW 2026). "
                  "They want to buy and they do not know how to evaluate. A governed, "
                  "auditable, measurable deployment is the answer to <i>their</i> stated "
                  "problem.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>The decision maker signs alone.</b> A managing "
                  "partner in a 40-person practice can commit £5k without a procurement "
                  "process.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>ROI is trivially measurable.</b> Practice "
                  "management software already counts overdue records requests, days to "
                  "response and jobs by status. You do not have to invent a metric — you "
                  "read theirs, before and after.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>It matches what NOVA can do <i>today</i>.</b> "
                  "Keyword retrieval over the firm's own documents with citations; email "
                  "and chat channels; recurring chasing; a human approval gate on anything "
                  "that leaves the building; an audit log a regulator would accept. No part "
                  "of the first offer needs semantic search, a hard cost ceiling, or a "
                  "field-validated channel you do not have.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>The market is big enough and nowhere near "
                  "saturated.</b> ~29,600–35,900 UK accounting businesses; adoption is "
                  "shallow across the whole economy (~1.6 technologies per adopter).",
                  S["bullet"]),
        Gap(2),
        Paragraph('<font color="#%s"><b>The honest counter-argument:</b></font> ICAEW is '
                  'publicly flagging over-reliance on AI as "the next big threat" to firms\' '
                  'reputation. That is a real objection and you will hear it. It is also '
                  'the best possible setup for NOVA\'s actual differentiator — approvals, '
                  'audit, per-agent permissions and citations are the answer to exactly '
                  'that fear. Lead with it rather than waiting to be hit with it.'
                  % WARNING.hexval()[2:], S["body"]),
    ], accent=GOOD, fill=PANEL_HI)]

    s += [Gap(4)]
    s += [table([
        ["Backup", "Vertical", "Why it is second, and when to switch to it"],
        ["<b>#2</b>", "<b>Property &amp; block management</b>",
         "Enormous repetitive tenant/leaseholder communication, statutory deadlines, and "
         "document volume. Scores nearly as well on workflow fit. It is second only because "
         "ROI is measured in response time rather than in a number the firm already tracks, "
         "and because the buyer is more fragmented. <b>Switch if</b> your first five "
         "accountancy conversations stall on professional-body caution."],
        ["<b>#3</b>", "<b>Recruitment agencies</b>",
         "Very high email and CV volume, owner-operated, fast decisions, comfortable with "
         "tooling. Third because the space is crowded with point AI tools already embedded "
         "in the dominant ATS, so you compete on “why not just use the Bullhorn feature” "
         "rather than on unmet need. <b>Switch if</b> you find a warm route in — this "
         "vertical rewards a referral far more than cold outreach."],
    ], [16 * mm, 40 * mm, W - 56 * mm], font_size=7.6)]

    s += [Gap(3), callout(
        "Two verticals to deliberately avoid first",
        "<b>FCA-regulated financial services</b> scores 4.4 and should be avoided until you "
        "have three case studies and a completed security questionnaire pack. The sales "
        "cycle includes a model-risk review you cannot currently pass — no field-validated "
        "provider, no production history. <b>Legal firms</b> score 5.2 for a related reason: "
        "the work is high-value but an error is a negligence claim, and BM25 keyword "
        "retrieval without semantic search is a weak fit for case research. Both become good "
        "markets later. Neither is a first customer.", "warn")]
    return s


# ---------------------------------------------------------------- PART 5


def part_5():
    s = section("05", "The first NOVA product offer", "One workflow, not a platform")

    s += [P(
        "Do not sell “AI agents”. A managing partner does not have a budget line for AI "
        "agents and cannot tell you whether one worked. Sell the removal of one specific, "
        "named, painful, measurable job.", "lede")]

    s += [Panel([
        Paragraph('<font color="#%s"><b>THE OFFER</b></font>' % ACCENT.hexval()[2:], S["kicker"]),
        Paragraph("NOVA Client Response &amp; Records Desk", S["h1"]),
        Gap(2),
        P("<b>The promise:</b> your practice stops losing partner and manager hours to "
          "chasing clients for records and answering the same routine questions — without "
          "anything reaching a client that a human did not approve, and with a log of every "
          "action your PI insurer can read."),
        Gap(1),
        P("<b>Deployed as three named workers inside your own AWS account:</b>"),
        Paragraph("<bullet>&#8226;</bullet><b>The Records Chaser</b> — knows which clients "
                  "owe what and by when, drafts the chase, escalates the ones that are "
                  "drifting.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>The Client Desk</b> — drafts replies to "
                  "routine client email from your own documented policies, and cites which "
                  "document it used.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>The Practice Librarian</b> — answers your "
                  "staff's “where do I find / what's our policy on” questions from the "
                  "firm's own handbook and procedures, with citations.", S["bullet"]),
    ], accent=ACCENT, fill=PANEL_HI)]

    s += [h2("5.1  Why these three, and not the exciting ones")]
    s += [P(
        "Each maps onto a capability that is at least <b>TESTED</b> today and needs no "
        "capability NOVA does not have. None of them touches bookkeeping entries, tax "
        "computations, filings or anything a professional body would call regulated advice "
        "— which is also what makes the first sale possible.")]
    s += [table([
        ["Worker", "NOVA capability it rides on", "Status"],
        ["Records Chaser", "Governed automations + channels + approvals + audit",
         status_chip("TESTED")],
        ["Client Desk", "Knowledge (FTS5 + citations) + channels + approvals",
         status_chip("LIVE LOCAL PROVEN")],
        ["Practice Librarian", "Knowledge only — no outbound channel at all",
         status_chip("LIVE LOCAL PROVEN")],
    ], [36 * mm, W - 74 * mm, 38 * mm])]

    s += [Gap(2), callout(
        "Start with the Librarian on day one",
        "It is the only one of the three with <b>no outbound channel</b>, so it cannot "
        "embarrass anybody. It produces visible value in week one, it builds the knowledge "
        "corpus the other two depend on, and it earns the trust you need before you ask a "
        "partner to let software draft a client email. Sequence matters more than scope.",
        "good")]

    s += [h2("5.2  Green / Yellow / Red — decided before deployment, not after")]
    s += [P(
        "This table goes in the contract and on the wall. It is the single most effective "
        "objection-handler you have, because it answers “what if it does something stupid” "
        "with a document rather than a reassurance. It is enforced by the compiled policy "
        "and the approvals queue, not by a prompt asking the model to be careful.")]

    green = [
        "Answer a staff question from the firm's own documents, with citations",
        "Draft a chase message into the queue for a human to release",
        "Summarise which clients are overdue and by how long",
        "Classify an inbound document by type and attach it to the right client",
        "Flag an approaching statutory deadline internally",
        "Log every action to the audit trail",
    ]
    yellow = [
        "Send any message to a client (email, chat, SMS)",
        "Second and third chase escalations",
        "Anything that names a fee, a deadline commitment or a service scope",
        "Adding a document to the knowledge base that will shape future answers",
        "Any reply on a channel marked externally-visible",
        "Creating or changing a recurring automation",
    ]
    red = [
        "Tax advice, computations or anything a client could act on as advice",
        "Filing anything with HMRC or Companies House",
        "Access to the accounting ledger, bank feeds or payment systems",
        "Anything touching money movement, ever",
        "Deciding to disengage or take on a client",
        "Handling a complaint, a dispute or anything with a legal dimension",
        "Anything where the firm's professional indemnity position is in play",
    ]
    for title, items, colour, note in [
        ("GREEN — the workforce acts alone", green, GOOD,
         "Enforced by: the agent's compiled policy allow-list. Logged, not queued."),
        ("YELLOW — a human approves before it happens", yellow, WARNING,
         "Enforced by: approval.required_for on the agent, plus per-channel approval "
         "policy. The action stops and queues; a named human releases it; both the "
         "intent and the release are audited."),
        ("RED — the workforce may not do this at all", red, CRITICAL,
         "Enforced by: tools.deny, which compiles into the runtime's own unconditional "
         "deny list — evaluated before any bypass mode. Plus: never grant the channel, "
         "never grant the corpus, never grant the toolset."),
    ]:
        body = [Paragraph(f'<font color="#{colour.hexval()[2:]}"><b>{title}</b></font>',
                          S["h3"])]
        body += [Paragraph(f"<bullet>&#8226;</bullet>{i}", S["bullet"]) for i in items]
        body += [Gap(1.5), Paragraph(f"<i>{note}</i>", S["cell_muted"])]
        s += [Panel(body, accent=colour, fill=PANEL_HI, pad=5 * mm), Gap(3)]

    s += [h2("5.3  The exact first deployment")]
    s += [Flow([
        ("Librarian only", "One corpus: the firm's handbook. No channel. Internal users only."),
        ("Add Client Desk", "Drafts only. Every send queued for approval."),
        ("Add Records Chaser", "First chase drafted, second and third escalate to a human."),
        ("Measure", "Against the firm's own practice-management numbers."),
    ], per_row=4)]
    s += [caption("Three to four weeks, in this order. Each stage has to be visibly working "
                  "before the next is switched on.")]

    s += [Gap(2), callout(
        "What must be true before day one that is not true today",
        "The Records Chaser depends on schedules actually firing, and §3.4 grades that "
        "<b>NOT IMPLEMENTED</b> — the cron ticker lives inside the runtime's gateway and no "
        "gateway process has ever been run in a NOVA deployment. Either prove that first "
        "(action 3 in §24), or scope the pilot to the Librarian and the Client Desk and "
        "trigger chases manually. <b>Do not sell the Chaser until you have watched a "
        "scheduled job fire.</b>", "risk")]
    return s


# ---------------------------------------------------------------- PART 6


def part_6():
    s = section("06", "Customer ROI model", "How to measure it, without inventing it")

    s += [P(
        "The entire commercial case rests on measuring something the customer already "
        "counts. Do not build a bespoke metric; read theirs. Every practice-management "
        "system in this market already tracks jobs by status, records requests, and days "
        "outstanding.", "lede")]

    s += [h2("6.1  The measurement method")]
    s += [P("<b>Two weeks of baseline before anything is switched on.</b> This is "
            "non-negotiable and it is the step that gets skipped. Without it, everything "
            "afterwards is an anecdote.")]
    s += [table([
        ["Metric", "Where it comes from", "How NOVA contributes the “after”"],
        ["Hours on the target workflow", "Time recording, or a two-week tally sheet",
         "Time recording after go-live, same categories"],
        ["Records requests outstanding &gt; 14 days", "Practice management software",
         "Same report, same filter"],
        ["Median days from request to receipt", "Practice management software", "Same report"],
        ["Median first-response time to client email", "Mail system reporting",
         "Same report"],
        ["Tasks completed by the workforce", "n/a", "NOVA Work screen — durable task board"],
        ["Automation rate (acted alone ÷ total)", "n/a",
         "NOVA audit log: committed actions with no approval record"],
        ["Human escalation rate", "n/a", "NOVA Approvals: queued ÷ total"],
        ["Error / correction rate", "Manual review of a sample",
         "Sample 30 outputs weekly and have a manager grade them. Do this honestly — it is "
         "the number the customer will ask about in month three."],
        ["Cost per workflow instance", "(loaded hourly rate × hours) ÷ instances",
         "Same calculation with the after-hours"],
        ["Revenue protected", "Fees at risk from capacity turn-away",
         "Only claim this if the firm can name the clients it turned away"],
    ], [42 * mm, 46 * mm, W - 88 * mm], font_size=7.6)]

    s += [h2("6.2  The calculation")]
    s += [Panel([
        Paragraph("Monthly labour value recovered", S["h3"]),
        Paragraph("= (baseline hours − after hours) × loaded hourly rate", S["mono"]),
        Gap(2),
        Paragraph("Loaded hourly rate", S["h3"]),
        Paragraph("= (salary × 1.30 for employer NI, pension, overhead) ÷ 1,650 working hours",
                  S["mono"]),
        Gap(2),
        Paragraph("Net monthly benefit", S["h3"]),
        Paragraph("= labour value recovered − NOVA monthly fee − AWS + model spend", S["mono"]),
        Gap(2),
        Paragraph("Payback period (months)", S["h3"]),
        Paragraph("= implementation fee ÷ net monthly benefit", S["mono"]),
    ], fill=PANEL_HI, accent=ACCENT)]
    s += [caption("Use the customer's own salary figures. If they will not share them, UK "
                  "market data for an accounts assistant sits around £24k–£27k depending on "
                  "source (ONS ASHE-derived and commercial salary trackers, 2026) — but "
                  "their number beats any published one, and asking for it is itself a "
                  "qualifying question.")]

    s += [h2("6.3  An illustrative scenario")]
    s += [callout(
        "Everything in this scenario is an assumption",
        "The hours below are <b>invented for the purpose of showing the arithmetic</b>. "
        "NOVA has no customers and therefore no results. Use this structure with the "
        "customer's own baseline numbers filled in; never present these figures as "
        "evidence, and never put them on a slide without this warning attached.", "warn")]
    s += [img("economic-impact.png")]
    s += [caption("ILLUSTRATIVE ASSUMPTION. A hypothetical 30-person practice. Not measured, "
                  "not a benchmark, not a promise.")]

    s += [table([
        ["Line", "Illustrative value", "Where a real number would come from"],
        ["Baseline hours/month on the five workflows", "138", "Two-week tally × 2"],
        ["Assumed hours after", "42", "Same tally, post go-live"],
        ["Hours recovered", "<b>96</b>", "Subtraction"],
        ["Assumed loaded rate", "£21/hour", "Firm's own payroll ÷ 1,650"],
        ["Labour value recovered", "<b>£2,016/month</b>", "96 × £21"],
        ["Less NOVA monthly fee", "−£950", "§7"],
        ["Less AWS + model spend", "−£250 (assumed)", "Actual bill — see §18"],
        ["Net monthly benefit", "<b>£816</b>", ""],
        ["Implementation fee", "£4,500", "§7"],
        ["Payback", "<b>~5.5 months</b>", "4,500 ÷ 816"],
    ], [58 * mm, 34 * mm, W - 92 * mm], font_size=7.8)]
    s += [caption("ILLUSTRATIVE ASSUMPTION throughout. The purpose of this table is the "
                  "method and the shape of the argument, not the numbers.")]

    s += [Gap(2), callout(
        "The honest thing to say in the room",
        "“I can't tell you what you'll save, because you'd be our first customer in this "
        "sector and I'm not going to quote you someone else's number. What I will do is "
        "measure your baseline for two weeks before we switch anything on, and at day 60 "
        "we look at your own report together. If it hasn't moved, you've paid for a pilot "
        "and you stop.” — This wins more deals than a fabricated case study, and it is the "
        "only version you can actually defend.", "good")]
    return s


# ---------------------------------------------------------------- PART 7


def part_7():
    s = section("07", "Pricing", "What to charge, and what never to charge for")

    s += [h2("7.1  What the UK market is paying")]
    s += [P(
        "Published 2026 benchmarks for UK and international AI automation work cluster as "
        "below. Treat these as the shape of the market, not as a price list — most are "
        "published by agencies with an interest in the number.")]
    s += [table([
        ["Segment", "Implementation", "Monthly", "Source type"],
        ["Single scoped workflow, UK", "£1,000–£5,000", "£200–£800",
         "UK agency pricing surveys, 2026"],
        ["Connected multi-workflow build, UK", "£4,000–£12,000", "—",
         "UK agency pricing surveys, 2026"],
        ["Small business, 2–3 workflows", "$5,000–$25,000", "$1,000–$3,500/mo",
         "International agency benchmarks, 2026"],
        ["Mid-market AI stack + reporting", "—", "$4,000–$10,000/mo",
         "International agency benchmarks, 2026"],
        ["Custom agent development", "from $10,000, $50,000+ complex", "—",
         "International agency benchmarks, 2026"],
        ["UK marketing agency AI tooling", "—", "£350–£1,100 per seat/mo",
         "UK agency reporting, 2026"],
    ], [50 * mm, 34 * mm, 30 * mm, W - 114 * mm], font_size=7.6)]
    s += [caption("Sources listed in §25. These are self-published agency benchmarks and "
                  "should be read as directional. The prevailing 2026 structure is "
                  "consistent across all of them: setup fee + monthly retainer + usage.")]

    s += [h2("7.2  Recommended pricing")]
    tiers = [
        ("FOUNDING CUSTOMER", GOOD, "£4,500", "£950 / month",
         ["Customers 1–3 only. Price is explicitly time-limited and named as such.",
          "12-month term, price locked for 24 months.",
          "One workflow, up to three agents, one knowledge corpus, two channels.",
          "In exchange — and this is written into the agreement — a named case study, a "
          "reference call, and access to their baseline numbers.",
          "Customer pays AWS and model costs directly, in their own account.",
          "Fortnightly review call for the first 90 days."]),
        ("STANDARD SMB", ACCENT, "£7,500", "£1,450 / month",
         ["20–100 staff. The main line once you have two case studies.",
          "One workflow area, up to five agents, two corpora, three channels.",
          "12-month term, monthly thereafter, 60 days' notice.",
          "Quarterly optimisation review included.",
          "Additional workflow: £3,500 setup, +£450/month.",
          "Bespoke integration: quoted separately, never bundled."]),
        ("MID-MARKET", WARM, "£15,000–£25,000", "£2,800–£4,500 / month",
         ["100–400 staff, or multi-site.",
          "Unlimited agents within an agreed policy envelope; up to five corpora.",
          "Named delivery lead, monthly governance review, quarterly security review.",
          "Priority response SLA (business hours).",
          "Assumes a security questionnaire and a DPIA — budget two to six weeks for it."]),
        ("ENTERPRISE", ACCENT, "£40,000+", "£6,000+ / month",
         ["Do not chase this yet. You have no production history, no SOC 2, no on-call "
          "rota and no field validation. You will lose the deal in the security review and "
          "burn six months doing it.",
          "Revisit after three referenceable customers and a completed security pack.",
          "When you do: annual prepay, custom DPA, named architect, defined RTO/RPO."]),
    ]
    for name, colour, setup, monthly, points in tiers:
        body = [
            Paragraph(f'<font color="#{colour.hexval()[2:]}"><b>{name}</b></font>', S["kicker"]),
            Paragraph(f'<font size="17"><b>{setup}</b></font>  <font color="'
                      f'#{TEXT_MUTED.hexval()[2:]}" size="9">implementation</font>'
                      f'   ·   <font size="13"><b>{monthly}</b></font>', S["price"]),
            Gap(1),
        ]
        body += [Paragraph(f"<bullet>&#8226;</bullet>{p}", S["bullet"]) for p in points]
        s += [Panel(body, accent=colour, fill=PANEL_HI, pad=5 * mm), Gap(3)]

    s += [h2("7.3  Why the customer pays")]
    s += [P(
        "Tie every pound to a business outcome and never to consumption. The sentence that "
        "sells this tier is: <b>“the monthly fee is less than a third of one part-time "
        "administrator, and it is the only administrator whose every action is logged.”</b>")]
    s += [table([
        ["The fee buys", "Not", "Because"],
        ["A workflow removed from partners' desks", "“Access to AI”",
         "They can get access to AI for £20/seat. They cannot get this."],
        ["Governance they can show an insurer or a regulator",
         "A number of prompts",
         "Approvals, per-agent permissions, and a write-ahead audit log are the "
         "differentiator and the hardest thing to rebuild."],
        ["Deployment into their own account", "A seat on your platform",
         "Their data never leaves their AWS account. For this buyer that is worth more "
         "than any feature."],
        ["Someone accountable for it working", "Software",
         "At this size, the customer is buying an outcome with a person behind it."],
    ], [52 * mm, 34 * mm, W - 86 * mm], font_size=7.8)]

    s += [h2("7.4  What not to do")]
    s += [Panel([
        Paragraph('<font color="#%s"><b>PRICING MISTAKES THAT WILL COST YOU THE BUSINESS</b>'
                  '</font>' % CRITICAL.hexval()[2:], S["h3"]),
        Paragraph("<bullet>&#8226;</bullet><b>Never price per prompt, per token or per "
                  "message.</b> It makes the customer ration usage, which kills adoption, "
                  "which kills the renewal. It also makes your revenue a function of model "
                  "pricing you do not control.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>Never include unlimited model usage in a "
                  "fixed fee.</b> You have no hard spending ceiling — §3.8 — and one "
                  "runaway loop on an expensive model can exceed a month's fee. The "
                  "customer pays their own AWS and model bill, in their own account. This "
                  "is not a concession; it is the architecture, and it is also a selling "
                  "point.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>Never promise unlimited custom "
                  "development.</b> “And we'll build whatever integrations you need” has "
                  "destroyed more early-stage software companies than any competitor. Each "
                  "integration is quoted.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>Never discount without taking something "
                  "back.</b> A discount in exchange for a case study, a reference, a "
                  "longer term or a faster decision is strategy. A discount because they "
                  "pushed is a signal that the price was invented, and they will push again "
                  "at renewal.", S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>Never let the implementation fee disappear "
                  "into the monthly.</b> It funds the real cost of delivery and it "
                  "qualifies the buyer. A customer who will not pay to start will not stay.",
                  S["bullet"]),
        Paragraph("<bullet>&#8226;</bullet><b>Never quote before discovery.</b> You cannot "
                  "price a workflow you have not watched someone do.", S["bullet"]),
    ], accent=CRITICAL, fill=PANEL_HI)]
    return s


# ---------------------------------------------------------------- PART 8


def part_8():
    s = section("08", "Getting the first UK customer", "Where they are, and how to reach them")

    s += [P(
        "This section is deliberately concrete. Named sources, named criteria, named "
        "sequence. A first customer comes from about 100 well-chosen conversations, not "
        "from a marketing funnel you do not yet have traffic for.", "lede")]

    s += [h2("8.1  Where the prospects actually are")]
    s += [P(
        "Listed by how well each source lets you <b>filter</b>, which matters far more than "
        "how many records it holds. I have marked what I have verified exists versus what "
        "you should check yourself before relying on it.")]
    s += [table([
        ["Source", "What it gives you", "Verified?"],
        ["<b>ICAEW / ACCA / AAT member firm directories</b>",
         "Firm name, location, services, professional body. The highest-quality starting "
         "point because membership itself is a qualifier.",
         "Directories exist and are public. <b>Check current search filters yourself.</b>"],
        ["<b>Companies House</b>",
         "Free API and bulk data. Filter by SIC code 69201/69202 (accounting, bookkeeping "
         "and auditing), incorporation date, filing history, registered office. Gives you "
         "officer names — i.e. the partners.",
         "Public and free. SIC codes are the standard classification."],
        ["<b>LinkedIn Sales Navigator</b>",
         "The main working tool. Filter by industry (Accounting), headcount 11–200, "
         "geography, and seniority (Partner / Managing Partner / Practice Manager). Shows "
         "recent hiring, which signals capacity pressure.",
         "Paid. Budget ~£70–£100/month — the single best-value spend in this plan."],
        ["<b>Accountancy trade press</b> (AccountingWEB, Accountancy Age, Accountancy Today)",
         "Named practitioners talking publicly about capacity, MTD and AI. A partner who "
         "has written about being overloaded is a warm lead.", "Publications verified active in 2026."],
        ["<b>Accountex and regional practice events</b>",
         "The main UK accountancy technology event. Also regional ICAEW branch meetings, "
         "which are smaller and far easier to actually meet people at.",
         "Accountex London ran in 2026. <b>Check dates and cost yourself.</b>"],
        ["<b>Practice software ecosystems</b> (Xero, Sage, IRIS, TaxCalc partner lists)",
         "Firms already comfortable buying software. Partner directories are public.",
         "Directories exist. Terms of use vary — check before scraping."],
        ["<b>Referrals from your own network</b>",
         "Every business has an accountant. Ask yours. Ask your solicitor's. This is the "
         "highest-conversion channel by a wide margin and costs nothing.", "n/a"],
        ["<b>Local business networks</b> (chambers of commerce, BNI, Federation of Small "
         "Businesses)", "Accountants are heavily over-represented in these, because that is "
         "where they find clients. So it is where you find them.", "n/a"],
        ["Google Maps / local directories", "Coverage of independent practices by town. "
         "Useful for geographic completeness, weak on firm size.", "n/a"],
        ["Clutch, Upwork, agency directories", "<b>Largely irrelevant for this vertical.</b> "
         "Useful if you switch to #3 (recruitment) or digital agencies.", "n/a"],
    ], [46 * mm, W - 46 * mm - 44 * mm, 44 * mm], font_size=7.4)]

    s += [h2("8.2  Prospect criteria")]
    s += [table([
        ["Must have", "Strong signal", "Disqualify"],
        ["20–150 staff", "Recently hired or is advertising for admin/support roles",
         "Under 10 staff — no budget, owner does everything"],
        ["UK-based, single or few offices", "Public comment about capacity or MTD workload",
         "Top 50 firm — procurement, InfoSec, and an internal team"],
        ["Cloud practice software already (Xero / Sage / IRIS / Karbon)",
         "Has a practice manager or operations manager (a named process owner)",
         "No cloud software at all — the integration story collapses"],
        ["An identifiable partner or MD you can reach",
         "Website mentions client service or responsiveness as a differentiator",
         "Already has an internal AI/automation lead — you become a bake-off"],
        ["Client base of 100+ (so chasing volume is real)",
         "Growing: new office, new hires, acquisition",
         "Currently in a merger — nothing gets decided"],
    ], [(W) / 3 - 2, (W) / 3 - 2, (W) / 3 - 2], font_size=7.6)]

    s += [Gap(2), callout(
        "The qualifying question that saves you the most time",
        "“Roughly how many hours a week does your team spend chasing clients for "
        "records?” If they answer with a number, they have thought about it, it hurts, and "
        "you have a live opportunity. If they say “no idea, not much”, politely move on — "
        "you cannot sell measurable ROI to someone who does not measure. Ask it in the "
        "first five minutes of every call.", "good")]
    return s


# ---------------------------------------------------------------- PART 9


def part_9():
    s = section("09", "The first 100 prospects", "A list, a score, and a 30-day plan")

    s += [h2("9.1  The tracker")]
    s += [P("One spreadsheet. Build it by hand — the research <i>is</i> the qualification, "
            "and outsourcing it produces a list you cannot sell to.")]
    cols = [
        ("Company", "Legal name from Companies House"),
        ("Website", ""),
        ("Industry / niche", "e.g. 'owner-managed businesses', 'contractors', 'dental'"),
        ("Employees", "LinkedIn, sanity-checked against the website's team page"),
        ("Decision maker", "Named human"),
        ("Role", "Managing Partner / Partner / Practice Manager"),
        ("LinkedIn URL", ""),
        ("Email", "Firm's published address until you have a reason to use a personal one"),
        ("Pain hypothesis", "One sentence, specific to them — this is the whole email"),
        ("Existing software", "Xero / Sage / IRIS / Karbon / unknown"),
        ("AI maturity", "None / experimenting / tooling in place"),
        ("Priority", "A / B / C from the score below"),
        ("Contacted", "Date + which touch"),
        ("Response", "None / negative / neutral / positive"),
        ("Meeting", "Date"),
        ("Proposal", "Date + value"),
        ("Won / Lost", "+ one-line reason. This column is how you learn."),
    ]
    rows = [["Column", "What goes in it"]] + [[c, d] for c, d in cols]
    s += [table(rows, [44 * mm, W - 44 * mm], font_size=7.6)]

    s += [h2("9.2  Lead scoring")]
    s += [table([
        ["Points", "Signal"],
        ["+3", "20–150 staff"],
        ["+3", "Named, reachable partner or MD"],
        ["+2", "Cloud practice software confirmed"],
        ["+2", "Hiring admin/support, or publicly discussing capacity"],
        ["+2", "Warm route in (referral, mutual connection, met at an event)"],
        ["+1", "Active on LinkedIn — they will see your content"],
        ["+1", "Website emphasises responsiveness or client service"],
        ["−2", "No cloud software"],
        ["−3", "Top 50 firm, or an internal AI lead"],
        ["−5", "Under 10 staff"],
    ], [18 * mm, W - 18 * mm], font_size=7.8)]
    s += [table([
        ["Band", "Score", "Treatment"],
        ["<b>A</b>", "9+", "Personal research, multi-touch, phone. ~25 of the 100."],
        ["<b>B</b>", "5–8", "Templated-but-specific email plus LinkedIn. ~50."],
        ["<b>C</b>", "&lt;5", "One email. Do not spend more. ~25."],
    ], [16 * mm, 20 * mm, W - 36 * mm], font_size=7.8)]

    s += [h2("9.3  The 30-day plan")]
    s += [table([
        ["", "Focus", "Daily target", "End-of-week gate"],
        ["<b>Week 1</b>", "Research and list building. No outreach at all — resist this.",
         "20 fully-researched prospects/day incl. a specific pain hypothesis",
         "100 rows complete and scored. 25 A-band identified."],
        ["<b>Week 2</b>", "First-touch outreach and follow-up. Start with B-band to "
         "rehearse the message before you spend your A-band.",
         "25 first-touch emails + 10 LinkedIn connections + 5 calls",
         "All 100 contacted once. At least 40 second touches sent."],
        ["<b>Week 3</b>", "Conversations. Every reply gets a call offer within two hours.",
         "5 calls attempted, 2 discovery calls held, all follow-ups within 24h",
         "8–12 discovery calls held. 4–6 demos booked."],
        ["<b>Week 4</b>", "Demos and pilot proposals. Proposal within 24 hours of every "
         "demo, while it is still warm.",
         "2 demos/day, proposal out same day",
         "2–3 pilot proposals live. One verbal commitment."],
    ], [18 * mm, W - 18 * mm - 44 * mm - 40 * mm, 44 * mm, 40 * mm], font_size=7.4)]

    s += [Gap(3), img("funnel.png")]
    s += [caption("ILLUSTRATIVE PLANNING TARGET. Set from general cold-outreach norms "
                  "because NOVA has no conversion history of its own. Use it to size "
                  "activity, and replace every number with your own after the first 100.")]

    s += [Gap(2), callout(
        "If 100 prospects produce nothing",
        "That is information, not failure, and it means one of exactly three things: the "
        "message is wrong, the vertical is wrong, or the offer is wrong. Diagnose in that "
        "order — message is cheapest to change and is wrong most often. Do <b>not</b> "
        "respond by sending another 100 of the same email; that tells you nothing you did "
        "not already know.", "warn")]
    return s


# ---------------------------------------------------------------- PART 10


def part_10():
    s = section("10", "Marketing and advertising", "What to do, and what to skip")

    s += [h2("10.1  Organic — where nearly all your effort goes")]
    s += [table([
        ["Channel", "What to publish", "Cadence", "Why it works here"],
        ["<b>LinkedIn, founder account</b>", "Not the company page — nobody follows company "
         "pages. Build a personal audience of UK accountants.", "3×/week",
         "Your buyer is genuinely active here, and a partner will read a founder before a "
         "brand."],
        ["<b>The governance angle</b>", "“What happens when an AI agent gets it wrong, and "
         "who finds out”. Screenshots of a real approvals queue and a real audit log.",
         "1×/week", "ICAEW is publicly flagging over-reliance on AI as a reputational "
         "threat. You are selling the answer to a fear the profession has already named."],
        ["<b>Before/after workflow posts</b>", "One workflow, the hours it takes, what it "
         "looks like governed. No product screenshots for the first three lines.",
         "1×/week", "Concrete beats conceptual with this audience every time."],
        ["<b>Short video (60–90s)</b>", "Screen recording of the Control Center doing one "
         "real thing end to end. No music, no motion graphics.", "2×/month",
         "Proves it exists. Most of your competition at this stage is a landing page."],
        ["<b>Technical architecture posts</b>", "How single-tenant BYOC deployment works. "
         "IAM boundaries. Why no secret is stored.", "1×/month",
         "Reaches the technically-minded partner and anyone doing due diligence on you."],
        ["<b>Customer ROI stories</b>", "<b>Only once you have one.</b> Named firm, their "
         "numbers, their quote.", "when earned",
         "This is the single highest-value marketing asset you will ever produce — which is "
         "why the founding-customer discount buys it contractually."],
        ["<b>Open-source credibility</b>", "Be straight that NOVA governs the open-source "
         "Hermes runtime. Contribute upstream where you can.", "ongoing",
         "Honesty here is a differentiator. It also pre-empts the discovery, which is far "
         "worse if the buyer makes it themselves."],
    ], [38 * mm, W - 38 * mm - 22 * mm - 50 * mm, 22 * mm, 50 * mm], font_size=7.4)]

    s += [h2("10.2  Direct sales — where the first customer actually comes from")]
    s += [table([
        ["Motion", "Volume", "Notes"],
        ["Cold email, researched",
         "25/day in week 2, then 10/day sustained",
         "Four to six sentences. Their specific pain, one sentence on the mechanism, one "
         "ask. No attachment, no deck, no calendar link in the first email."],
        ["LinkedIn outreach", "10 connections/day",
         "Connect, then nothing for a week. Engage with their posts. Message only when you "
         "have a reason. Pitching in the connection request destroys the account."],
        ["Phone", "5 attempts/day on A-band only",
         "Still works in this market and almost nobody does it. Call the practice, ask for "
         "the practice manager. 20 seconds on why you called, then ask for 15 minutes."],
        ["Partner referrals", "ongoing",
         "Bookkeepers, fractional FDs, practice-software consultants, IT MSPs serving "
         "accountancy. They already have the relationship and no competing product. §15."],
        ["Events", "1–2 per quarter",
         "Regional ICAEW branch events over big trade shows. Ten real conversations beats "
         "400 badge scans, and costs a hundredth as much."],
    ], [34 * mm, 34 * mm, W - 68 * mm], font_size=7.6)]

    s += [h2("10.3  Paid advertising — the recommendation is no")]
    s += [Panel([
        Paragraph('<font color="#%s"><b>DO NOT RUN PAID ADS UNTIL CUSTOMER THREE</b></font>'
                  % CRITICAL.hexval()[2:], S["h3"]),
        P("This is a considered recommendation, not caution for its own sake. Paid "
          "acquisition converts attention into customers. You currently have no proven "
          "conversion, no case study, no measured ROI claim, and no landing page that has "
          "ever converted anyone. Spending on ads now buys traffic to a page that cannot "
          "close, and you will read the resulting zero as “ads do not work” when what it "
          "actually measured was the offer."),
        P("There is also an arithmetic problem specific to this market. LinkedIn Ads to a "
          "senior UK professional-services audience is among the most expensive inventory "
          "there is. At a realistic cost per qualified lead in that channel, and a "
          "first-deal value of roughly £4.5k setup plus £11.4k of first-year recurring, you "
          "would need a conversion rate from cold ad click to signed pilot that nobody "
          "achieves without an established brand and a case study. The maths does not "
          "close, and no budget fixes it."),
        Gap(2),
        Paragraph("When each channel becomes worth testing:", S["h3"]),
    ] + [Paragraph(f"<bullet>&#8226;</bullet>{t}", S["bullet"]) for t in [
        "<b>Google Search</b> — first, and only for high-intent phrases people actually "
        "type when they have already decided (“ai automation for accountancy practice uk”). "
        "Volume will be tiny, which is the point: tiny and intent-loaded is exactly what you "
        "want. Start at <b>£300–£500/month</b> for one month and judge on qualified "
        "conversations, not clicks.",
        "<b>LinkedIn Ads</b> — after two case studies, and then only as <b>retargeting</b> "
        "of people who already visited. Cold LinkedIn prospecting ads are for companies with "
        "brand recognition. Budget <b>£500–£1,000/month</b> when you get there.",
        "<b>Retargeting generally</b> — the first paid spend that makes sense, because it is "
        "the only one aimed at people who have already shown intent. Even so it needs "
        "meaningful traffic first.",
        "<b>Meta Ads</b> — <b>no.</b> Wrong audience, wrong intent, wrong buying mode for a "
        "£15k+ annual B2B professional-services purchase.",
    ]], accent=CRITICAL, fill=PANEL_HI)]

    s += [Gap(3), callout(
        "Where that budget should go instead",
        "LinkedIn Sales Navigator (~£70–£100/month) plus an email-verification tool plus "
        "your own time. For the price of one month of LinkedIn Ads you can research and "
        "contact several hundred named firms with a message written for each. At this stage "
        "that converts and advertising does not.", "good")]
    return s


# ---------------------------------------------------------------- PART 11


def part_11():
    s = section("11", "The NOVA message", "What to say, in every length")

    s += [P("Written against what NOVA actually does. Nothing here claims field validation, "
            "semantic search, spending caps, or a channel that has not been connected.",
            "lede")]

    def block(label, text, note=""):
        body = [Paragraph(f'<font color="#{ACCENT.hexval()[2:]}"><b>{label}</b></font>',
                          S["kicker"]),
                Paragraph(text, S["body"])]
        if note:
            body.append(Paragraph(f"<i>{note}</i>", S["cell_muted"]))
        return Panel(body, fill=PANEL_HI, accent=ACCENT, pad=5 * mm)

    s += [block("ONE-LINE PITCH",
                "<font size='13'><b>NOVA puts a governed AI workforce inside your business "
                "— in your own cloud, connected to your systems, and answerable to your "
                "rules.</b></font>")]
    s += [Gap(3)]

    s += [block("30-SECOND PITCH",
                "“Most firms have tried AI and got a chat assistant — it drafts things, and "
                "the admin comes straight back. NOVA is different: it deploys named AI "
                "workers <i>inside your own AWS account</i> that read your documents, work "
                "your channels, and handle recurring jobs like chasing clients for records. "
                "Every worker has a job description you write, permissions you set, and "
                "anything consequential stops for a human to approve. Every action is "
                "logged with who authorised it. Your data never leaves your account.”",
                "Roughly 75 words. Time it — 30 seconds is shorter than people think.")]
    s += [Gap(3)]

    s += [block("60-SECOND PITCH",
                "“A 40-person practice loses something like two working days a week to "
                "chasing clients for records and answering the same routine questions. "
                "You've probably tried AI for it. It helped with drafting and then the admin "
                "came straight back, because the assistant can't see your systems and can't "
                "be trusted to act.<br/><br/>"
                "NOVA deploys AI workers into your own AWS account — your infrastructure, "
                "your data, your encryption key. Each one has a job description you write "
                "and permissions you set. The Records Chaser knows who owes what and drafts "
                "the chase. The Client Desk answers routine client email from your own "
                "documented policies and cites which document it used. The Practice "
                "Librarian answers your team's questions from your handbook.<br/><br/>"
                "Anything that reaches a client stops and waits for a human to release it. "
                "Every action — including every refusal — is written to an audit log with "
                "the person who authorised it. That's the part your PI insurer will care "
                "about.<br/><br/>"
                "We start with a 60-day paid pilot. Two weeks measuring your baseline first, "
                "so at the end we're looking at your numbers, not mine.”")]
    s += [Gap(3)]

    s += [table([
        ["Asset", "Copy"],
        ["<b>Website headline</b>",
         "<font size='12'><b>Your AI workforce. Your cloud. Your rules.</b></font>"],
        ["<b>Website subheadline</b>",
         "NOVA deploys governed AI workers into your own AWS account — connected to your "
         "documents, your channels and your recurring work, with human approval on anything "
         "that matters and a full audit trail on everything else."],
        ["<b>LinkedIn company description</b>",
         "NOVA is an enterprise AI workforce platform for professional services firms. We "
         "deploy governed AI agents into your own cloud account, connected to your knowledge, "
         "systems and communication channels — with per-agent permissions, human approval on "
         "consequential actions, and a complete audit trail. Built on the open-source Hermes "
         "agent runtime. Your data never leaves your infrastructure."],
        ["<b>Founder explanation</b> (for “so what do you do?”)",
         "“I build the layer that makes AI agents safe to actually deploy in a business. The "
         "agent technology is open source and it's good. What's missing is everything a real "
         "company needs before it lets software act on its behalf — permissions, approvals, "
         "an audit trail, and running it in your own infrastructure instead of someone "
         "else's. That's what I build. Right now I'm working with accountancy practices, "
         "because they're drowning and they're regulated, which is the exact combination "
         "this is for.”"],
        ["<b>Sales presentation opening</b>",
         "“Before I show you anything — roughly how many hours a week does your team lose "
         "to chasing clients for records? [listen] And what happens today when a client "
         "just doesn't respond? [listen] Right. Let me show you what that looks like when "
         "it's handled, and I'll be specific about what this can't do as well as what it "
         "can.”"],
    ], [40 * mm, W - 40 * mm], font_size=8)]

    s += [h2("11.1  Words that are banned, and what to say instead")]
    s += [table([
        ["Never say", "Say instead", "Because"],
        ["Revolutionary / game-changing / cutting-edge / next-generation",
         "Nothing. Delete the sentence.",
         "These signal that you have no specific claim. A partner reads them as noise."],
        ["“Fully autonomous”", "“Acts alone on what you've approved it to act alone on”",
         "Autonomy is the fear, not the benefit, for this buyer."],
        ["“Replaces your admin team”", "“Gives your team back the hours they spend chasing”",
         "The buyer has a staffing shortage, not a staffing surplus. 95% of AI-using SMEs "
         "report no workforce reduction (BCC 2026) — so it is also untrue."],
        ["“Understands your documents”", "“Searches your documents and cites what it used”",
         "It is BM25 keyword retrieval. Say what it is; the citation is the actual benefit."],
        ["“Enterprise-grade security”", "Name the controls: own account, own KMS key, no "
         "inbound ports, SSM-only access, IAM permissions boundary, audit log.",
         "Specifics are checkable. The phrase is not."],
        ["“It just works”", "“Here is the 60-day pilot and here is what we measure”",
         "Nobody at this price point believes the first one."],
    ], [44 * mm, 52 * mm, W - 96 * mm], font_size=7.6)]
    return s


# ---------------------------------------------------------------- PART 12


def part_12():
    s = section("12", "Website structure", "Ten pages, and what goes on each")

    s += [h2("12.1  The homepage")]
    s += [Flow([
        ("Hero", "Headline, subheadline, one CTA"),
        ("Problem", "The hours, named"),
        ("AI Workforce", "The three workers"),
        ("How it works", "The four layers"),
        ("Security", "Your account, your key"),
    ], per_row=5)]
    s += [Gap(2), Flow([
        ("Integrations", "Channels, MCP, knowledge"),
        ("ROI", "The method, not a promise"),
        ("Use cases", "One per worker"),
        ("Demo", "Video, 90 seconds"),
        ("CTA", "Book a 15-minute call"),
    ], per_row=5)]
    s += [caption("The order matters. Security sits above integrations because for this "
                  "buyer it is the first objection, not the last.")]

    s += [Gap(3), table([
        ["Section", "What it must contain", "What it must not"],
        ["<b>Hero</b>", "The headline from §11, a one-line subhead, and exactly one CTA: "
         "“Book a 15-minute call”. A still of the real Control Center.",
         "A carousel. A second CTA. A stock photo of a robot."],
        ["<b>Problem</b>", "The specific hours: chasing records, routine client email, "
         "internal questions. Written so a partner recognises their own week.",
         "Statistics about the AI market. They do not care."],
        ["<b>AI Workforce</b>", "The three named workers, one card each, with the one "
         "sentence each does.", "A feature grid of twelve capabilities."],
        ["<b>How NOVA works</b>", "The four-layer diagram from §2, simplified. Name Hermes "
         "honestly in one line.", "Architecture depth. Link to a separate page for that."],
        ["<b>Security</b>", "Your AWS account. Your KMS key. No inbound ports. SSM-only "
         "access. No credential stored by us. Audit log of every action.",
         "“Bank-level security”. A padlock icon. Compliance badges you do not hold."],
        ["<b>Integrations</b>", "Honest categories: 22 channel platforms available, 65 MCP "
         "servers, your own documents. A note that each is enabled per agent.",
         "<b>A logo wall.</b> Zero channels are field-validated — a Slack logo is a claim "
         "you cannot currently support."],
        ["<b>ROI</b>", "The measurement method from §6. “We measure your baseline for two "
         "weeks before we switch anything on.”",
         "Invented percentages. Any figure you cannot attribute."],
        ["<b>Use cases</b>", "One page per worker, each with the Green/Yellow/Red table.",
         "Generic 'AI for business' content."],
        ["<b>Demo</b>", "A 90-second screen recording of one real thing, end to end.",
         "A motion-graphics product film. It reads as vapourware."],
        ["<b>CTA</b>", "Calendar link. Name, firm, size, one line on the workflow.",
         "A 9-field form. A chatbot."],
    ], [30 * mm, (W - 30 * mm) * 0.52, (W - 30 * mm) * 0.48], font_size=7.6)]

    s += [h2("12.2  The other nine pages")]
    s += [table([
        ["Page", "Job it does"],
        ["<b>How it works</b>", "For the person who has to explain it internally. The full "
         "architecture, the deployment model, what apply does, what an agent is."],
        ["<b>AI Workforce</b>", "The capability catalogue in customer language. Be graded — "
         "an “available / in pilot” column earns more trust than it costs."],
        ["<b>Governance</b>", "<b>Your strongest page. Build it first after the homepage.</b> "
         "Permissions, approvals, audit, Green/Yellow/Red, screenshots of a real audit log."],
        ["<b>Integrations</b>", "Categories and mechanism, not logos. Explain that a "
         "connection is granted to a named agent and revoked in one click."],
        ["<b>Security</b>", "For the buyer's IT person or MSP. Deployment topology, IAM "
         "model, encryption, where data lives, what you can and cannot see. Offer to answer "
         "a security questionnaire — most competitors at this size will not."],
        ["<b>Industries</b>", "One page: accountancy. Ten thin pages are worse than one "
         "page that is obviously written by someone who understands the work."],
        ["<b>Pricing</b>", "<b>Publish the implementation fee and the monthly.</b> Your "
         "buyer is a professional-services firm; they respect transparent pricing and "
         "“contact us” loses you the ones who would have bought."],
        ["<b>Case studies</b>", "Empty until you have one. An honest “our first pilots are "
         "running now” beats three invented logos, which is the fastest way to lose a "
         "partner's trust permanently."],
        ["<b>Contact / demo</b>", "Calendar. Four fields. Nothing else."],
    ], [32 * mm, W - 32 * mm], font_size=7.8)]
    return s
