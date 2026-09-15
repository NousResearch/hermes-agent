"""Parts 13-25: demo, pilot, partners, best practice, AWS deployment, cost, checklist,
roadmap, revenue, positioning, risks, next actions, sources."""

from __future__ import annotations

from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Spacer

from layout import bullets, caption, h2, h3, img, section
from theme import (ACCENT, AQUA, CONTENT_W, CRITICAL, GOOD, PANEL_HI, S, SERIOUS, TEXT_MUTED,
                   WARM, WARNING, Checklist, Flow, Gap, P, Panel, Rule, Timeline, callout,
                   status_chip, table)

W = CONTENT_W


# ---------------------------------------------------------------- PART 13


def part_13():
    s = section("13", "The 15-minute sales demo", "A storyline, not a tour")

    s += [P(
        "The demo's job is to make a partner believe two things: that this removes a job "
        "they hate, and that it cannot embarrass them. Everything that does not serve one "
        "of those is cut, including most of the architecture you are proud of.", "lede")]

    s += [callout(
        "Three rules for the whole 15 minutes",
        "<b>1. Use their language.</b> “Records request”, not “task”. “Client”, not "
        "“end user”. “Partner sign-off”, not “approval workflow”. "
        "<b>2. Never say the word “agent” in the first eight minutes.</b> Say “a worker "
        "called the Records Chaser”. "
        "<b>3. Name one limitation out loud, unprompted.</b> It buys more credibility than "
        "any feature, and they will find it anyway.", "note")]

    slots = [
        ("0–2", "The business problem", GOOD,
         ["Screen off. You are talking, not showing.",
          "“Before I show you anything — how many hours a week does your team lose to "
          "chasing records?” Then listen properly.",
          "Play their number back with the annual cost attached. “So that's roughly "
          "£X a year of partner and manager time on chasing.”",
          "“And the bit I hear most is that it isn't the time, it's that it never ends. "
          "Is that fair?”"]),
        ("2–5", "The Control Center", ACCENT,
         ["Open on <b>Overview</b>. Their firm's name and their brand colour, because "
          "white-labelling is real and it lands instantly.",
          "“This is your workforce. Three workers. This is what each one has done this "
          "week.”",
          "Click into <b>Work</b>. Real tasks, real states.",
          "Do not explain the architecture. They have not asked."]),
        ("5–8", "The workforce itself", ACCENT,
         ["Open the <b>Records Chaser</b> profile.",
          "<b>Soul</b> tab: “this is its job description — in plain English, and you can "
          "edit it.” Change a line and save it in front of them. This is the moment most "
          "demos are won.",
          "<b>Permissions</b> tab: “this is everything it's allowed to touch. Nothing else "
          "is on this list, so nothing else can happen.”",
          "“And if you want a fourth worker, you create one here.” Do not actually create "
          "it — you will burn 90 seconds."]),
        ("8–10", "Knowledge and connections", AQUA,
         ["<b>Knowledge</b>: upload one of <i>their</i> documents live if they have sent "
          "one. Otherwise a realistic sample handbook.",
          "Ask a question against it. Show the answer <b>with the citation</b>.",
          "“It tells you which document it used. If you change the document, the answer "
          "changes. It isn't making it up from the internet.”",
          "<b>Say the limitation here:</b> “This is keyword search over your documents, not "
          "the fuzzy semantic kind. It's excellent when your documents use your firm's own "
          "vocabulary, which yours do. I'd rather tell you that now than have you find it "
          "in week three.”",
          "<b>Channels</b>: show the list and the per-agent grant. Do <b>not</b> claim any "
          "specific platform is live unless you have connected it for them."]),
        ("10–12", "Governance — the part that closes it", WARM,
         ["<b>Approvals</b>. “This client email is drafted and it is <i>not</i> going "
          "anywhere until someone here releases it.” Release one.",
          "<b>Activity</b>. “Every refusal and every escalation, with who authorised it "
          "and when.”",
          "Show the Green/Yellow/Red table. “We agree this before we deploy anything, and "
          "it's in the contract.”",
          "“Your PI insurer and your professional body are going to ask what happens when "
          "it gets something wrong. This is the answer.”"]),
        ("12–14", "Recurring work", ACCENT,
         ["<b>Automations</b>. “The Monday-morning chase runs on a schedule. Here it is, "
          "here's when it next runs, here's what it did last time.”",
          "“And it can't be given a new standing instruction without going through the "
          "same permission checks as everything else.”",
          "<b>Say the limitation here too</b> if the scheduler is not yet field-proven in "
          "your deployment: “Scheduling is the newest part of this and I'll be proving it "
          "in your environment in week one of the pilot rather than claiming it today.”"]),
        ("14–15", "ROI and the ask", GOOD,
         ["Back to their number from minute one.",
          "“You said about 30 hours a week. We measure that properly for two weeks before "
          "we switch anything on, so at day 60 we're looking at your report, not my slide.”",
          "<b>The ask:</b> “I'd like to run a 60-day pilot. £4,500 to set it up, £950 a "
          "month. You pay AWS directly — it'll be a few hundred a month in your own "
          "account. If the numbers haven't moved at day 60, we stop and you've paid for a "
          "pilot.”",
          "Then stop talking."]),
    ]
    for when, title, colour, beats in slots:
        body = [Paragraph(
            f'<font color="#{colour.hexval()[2:]}"><b>MINUTE {when}</b></font>  '
            f'<font color="#{TEXT_MUTED.hexval()[2:]}">|</font>  <b>{title}</b>', S["h3"])]
        body += [Paragraph(f"<bullet>&#8226;</bullet>{b}", S["bullet"]) for b in beats]
        s += [Panel(body, accent=colour, fill=PANEL_HI, pad=4.6 * mm), Gap(2.6)]

    s += [Gap(2), callout(
        "What to cut when you run out of time — and you will",
        "Cut minutes 12–14 (automations) before anything else. Never cut 10–12 "
        "(governance): it is the section that differentiates you from every other person "
        "who has emailed this partner about AI this year.", "warn")]
    return s


# ---------------------------------------------------------------- PART 14


def part_14():
    s = section("14", "The first customer pilot", "60 days, with a defined end")

    s += [P(
        "A pilot with no end date becomes a free deployment. This one has a fixed length, "
        "a fixed scope, an agreed definition of success written before it starts, and a "
        "decision at the end.", "lede")]

    s += [Timeline([
        ("WEEK 0", "Discovery and baseline",
         "Watch the workflow being done. Two-week baseline measurement starts now, before "
         "anything is installed. Agree Green/Yellow/Red. Name the success criteria and the "
         "one person on their side who owns it.", GOOD),
        ("WEEK 1", "Infrastructure and integration",
         "AWS account access, Terraform apply, image push, tenant bundle, principals, model "
         "credentials. Knowledge corpus loaded from their handbook. Nothing client-facing "
         "yet.", ACCENT),
        ("WEEK 2", "Agent configuration",
         "The Practice Librarian goes live to internal staff only. Personas written with "
         "them, not for them. Policy compiled and verified by attempting a denied action in "
         "front of them.", ACCENT),
        ("WEEK 3", "Controlled deployment",
         "Client Desk in draft-only mode — every message queued for approval. Records "
         "Chaser triggered manually, not scheduled, until the scheduler is proven in their "
         "environment.", WARM),
        ("WEEK 4–6", "Measurement and widening",
         "Approval rate should be falling as the drafts get better. Widen Green as trust "
         "earns it. Weekly 30-minute review with their owner. Sample 30 outputs a week and "
         "grade them honestly.", AQUA),
        ("DAY 60", "ROI review and decision",
         "Their numbers, their report, side by side with the baseline. Three outcomes: "
         "convert to standard, extend once with a named reason, or stop. Agree which before "
         "you walk in.", GOOD),
    ])]

    s += [h2("14.1  Success criteria")]
    s += [callout(
        "These are proposed pilot targets, not predictions",
        "NOVA has no customers, so no target here is derived from experience. Set them "
        "<i>with</i> the customer in week 0 against their own baseline, write them down, and "
        "sign them. A target you set unilaterally is one they will dispute at day 60.",
        "warn")]
    s += [table([
        ["Category", "Proposed measure", "Proposed target", "Why this one"],
        ["Hours saved", "Hours/month on the target workflow vs baseline", "−40% or better",
         "The headline. If this moves, nothing else matters much."],
        ["Response time", "Median first response to routine client email", "−50%",
         "Visible to their clients, so it is felt as well as measured."],
        ["Workflow completion", "Records requests closed within 14 days", "+25pp",
         "Their existing report, unchanged — no argument about methodology."],
        ["Escalation rate", "Share of actions needing human release", "Falling week on week",
         "The trajectory matters more than the level. Flat means it is not learning the "
         "firm's voice."],
        ["Error rate", "Outputs a manager marks wrong, from a 30/week sample",
         "&lt;5% by week 4", "Measure it honestly or it will be measured for you by a "
         "client complaint."],
        ["Adoption", "Staff using the Librarian weekly", "&gt;60% of fee earners",
         "Unused software does not renew, however good the hours number is."],
        ["Cost", "Total AWS + model spend", "Within agreed ceiling",
         "There is no hard cap in the product — see §3.8. Watch it manually and weekly."],
    ], [26 * mm, 46 * mm, 30 * mm, W - 102 * mm], font_size=7.6)]

    s += [h2("14.2  What kills a pilot")]
    s += [table([
        ["Failure", "Prevention"],
        ["No baseline measured", "Week 0 is not optional. Refuse to start without it."],
        ["Scope grew", "One workflow. Everything else goes on a written 'phase 2' list — "
         "which also becomes the upsell."],
        ["No owner on their side", "Name one person in week 0. If they cannot name one, "
         "they are not ready."],
        ["First client-facing output was wrong and nobody saw it",
         "Draft-only for the whole of week 3. No exceptions, however good it looks."],
        ["Champion left / got busy", "Two contacts from day one, and the partner in the "
         "weekly review even if they say they do not need to be."],
        ["The scheduler never fired", "Prove it in week 1 in <i>their</i> environment, "
         "before anything depends on it."],
    ], [56 * mm, W - 56 * mm], font_size=7.8)]
    return s


# ---------------------------------------------------------------- PART 15


def part_15():
    s = section("15", "The UK sales partner model", "Commission-only, structured properly")

    s += [P(
        "Published 2026 B2B benchmarks put new-business SaaS commission around 8–12% of "
        "new ARR, renewals at 2–8%, and pure-commission independent representatives "
        "materially higher — commonly quoted in the 15–30% range to compensate for carrying "
        "all the risk. The structure below sits inside those norms while protecting cash "
        "flow and, more importantly, protecting the customer relationship.", "lede")]

    s += [h2("15.1  Recommended structure")]
    s += [table([
        ["Term", "Recommendation", "Reasoning"],
        ["<b>Status</b>", "Self-employed contractor, written agreement. Not an employee.",
         "No PAYE, no notice period, no employment risk. Have a UK employment solicitor "
         "check it — misclassification is expensive."],
        ["<b>Base</b>", "None initially. Revisit at customer five.",
         "You cannot fund a salary pre-revenue, and a commission-only partner who will not "
         "take the risk does not believe in the product."],
        ["<b>Implementation commission</b>", "<b>20%</b> of the setup fee",
         "≈£900 on a £4,500 founding deal. Paid on <b>cleared funds</b>, never on signature."],
        ["<b>Year-1 recurring</b>", "<b>15%</b> of monthly revenue, months 1–12",
         "≈£143/month on a £950 deal. Sits in the independent-rep band and rewards "
         "landing a customer who stays."],
        ["<b>Year-2 recurring</b>", "<b>7%</b> for as long as the customer is live <i>and</i> "
         "the partner still owns the relationship",
         "Inside the 2–8% renewal norm. Drops to 0% if account management moves to you."],
        ["<b>Accelerator</b>", "Add 5pp on implementation after the 4th customer in a "
         "12-month window", "Rewards a repeatable motion rather than one lucky deal."],
        ["<b>Payment timing</b>", "Monthly in arrears, within 14 days of the customer's "
         "cleared payment", "Never pay commission on revenue you have not banked."],
        ["<b>Clawback</b>", "Full clawback if the customer cancels or refunds within 90 days",
         "Stops a partner selling a deal that was never going to stick."],
        ["<b>Lead attribution</b>", "Written register. A lead is the partner's for <b>90 "
         "days</b> from first registration; inbound leads to you are yours unless you assign "
         "them.", "Attribution disputes end partnerships. Decide it on day one, in writing."],
        ["<b>Customer ownership</b>", "<b>Yours, absolutely.</b> Contract, data, account and "
         "relationship are with your company.",
         "Non-negotiable. A partner who wants to own the customer is a reseller, which is a "
         "different agreement entirely."],
        ["<b>Territory</b>", "UK. Optionally a named vertical.",
         "Do not fragment a market this size geographically."],
        ["<b>Exclusivity</b>", "<b>None for the first 6 months.</b> Then optional, "
         "conditional on a quota.",
         "Exclusivity given before performance is proven is how you lose a market to one "
         "person's calendar."],
        ["<b>Termination</b>", "30 days either side. Commission on already-closed customers "
         "continues for 6 months after termination, then stops.",
         "Fair, finite, and it keeps the exit clean."],
        ["<b>Confidentiality</b>", "Mutual NDA. Explicitly covers customer names, the "
         "repository, pricing floors and the capability gaps in §23.",
         "They will see the honest limitations. That is deliberate and it must be protected."],
        ["<b>Authority to quote</b>", "May quote the published tiers only. Any discount, "
         "any custom scope, any term change needs your written approval.",
         "A partner discounting to close costs you the price anchor for every future deal."],
        ["<b>No unauthorised technical promises</b>",
         "Written list of what may and may not be claimed, derived from §3 and §23. "
         "Breach is a termination event.",
         "<b>The single most important clause.</b> A commission-only partner is structurally "
         "incentivised to promise anything. See the box below."],
    ], [38 * mm, 54 * mm, W - 92 * mm], font_size=7.4)]

    s += [Gap(3), callout(
        "The capability schedule — attach it to the agreement",
        "Give the partner a one-page annex listing exactly what may be claimed: governed "
        "agents, per-agent permissions, human approval, audit log, deployment in the "
        "customer's own AWS account, keyword document search with citations, 22 available "
        "channel platforms. And exactly what may <b>not</b>: any named channel as “live”, "
        "semantic or AI search, a hard spending cap, scheduled execution, multi-tenancy, "
        "any compliance certification, and any customer result. Update it every time §3 "
        "moves a rung. A promise your partner makes is a promise you have to deliver.",
        "risk")]

    s += [h2("15.2  When to hire a full-time salesperson")]
    s += [table([
        ["Do NOT hire until all of these are true", "Why"],
        ["3+ paying customers, at least 2 referenceable", "You are selling a proven thing, "
         "not testing whether it sells."],
        ["A repeatable sales process — the same demo and objections work twice",
         "Otherwise you are hiring someone to discover your process for you, at salary."],
        ["The <b>founder</b> has personally closed at least 3", "You cannot hire out a motion "
         "you have not run. You also cannot coach it or interview for it."],
        ["Deployment is repeatable — a second customer deployed without heroics",
         "Sales that outrun delivery burn the reputation you are selling."],
        ["12 months of the loaded cost in the bank", "UK B2B sales hires take 3–6 months to "
         "produce. Loaded cost is salary + ~30% for NI, pension and overhead."],
        ["A pipeline they can work from day one", "A new hire who must also build the list "
         "will take nine months, not six."],
    ], [66 * mm, W - 66 * mm], font_size=7.8)]
    s += [caption("Commission benchmarks: 2026 B2B and SaaS commission surveys (Everstage, "
                  "CaptivateIQ, Fullcast) — see §25. Interpret as market norms, not as a "
                  "published standard.")]
    return s


# ---------------------------------------------------------------- PART 16


def part_16():
    s = section("16", "Commercial best practices", "The rules, in five areas")

    groups = [
        ("PRODUCT", ACCENT, [
            "<b>Sell outcomes, not agents.</b> “Chasing handled” has a budget line. “AI "
            "agents” does not.",
            "<b>Start narrow and stay narrow.</b> One workflow, three workers. Everything "
            "else goes on a written phase-2 list — which is also your upsell pipeline.",
            "<b>Prove ROI with their numbers.</b> Never present a figure you generated.",
            "<b>Do not build ahead of a customer.</b> Every hour spent on a feature nobody "
            "has asked for is an hour not spent on field validation, which is the actual "
            "blocker.",
            "<b>Never promise a capability the repository cannot support.</b> §3 is the "
            "source of truth. When it moves a rung, the sales material moves with it — not "
            "before.",
            "<b>Say one limitation out loud in every sales conversation.</b> It is the "
            "cheapest credibility available and it inoculates you against discovery later.",
        ]),
        ("SECURITY", GOOD, [
            "<b>Least privilege, always.</b> The runtime IAM role carries no customer-data "
            "permissions and sits under a permissions boundary. Keep it that way even when "
            "a customer offers broader access to make an integration easier.",
            "<b>Customer-owned credentials.</b> NOVA never stores a secret. The profile "
            "<font face='Courier' size='7'>.env</font> is the customer's file and survives "
            "apply. Do not build a feature that changes this.",
            "<b>No secrets in images, ever.</b> Eight checks enforce it in the build. If a "
            "customer asks you to bake a key in, the answer is no.",
            "<b>One tenant, one deployment.</b> Until multi-tenancy exists — and it does "
            "not — this is the isolation guarantee. Do not co-locate two customers.",
            "<b>Audit everything consequential.</b> Intent before, outcome after, with the "
            "authenticated human. This is the product's spine, not a feature.",
            "<b>Human approval on anything that leaves the building.</b> Widen Green only "
            "when measured accuracy earns it, never because it is slow.",
            "<b>Never grant an agent broad access to the customer's AWS account.</b> Each "
            "integration is a separate role assumed with an ExternalId.",
        ]),
        ("SALES", WARM, [
            "<b>Do not lead with technology.</b> Eight minutes before the word “agent”.",
            "<b>Quantify the pain in the first five minutes</b> or disqualify. A prospect "
            "who cannot estimate the hours cannot evaluate the result.",
            "<b>Sell to the partner, not the office manager.</b> The office manager is your "
            "champion and cannot sign.",
            "<b>Always a paid pilot.</b> Free pilots do not get resourced, do not get "
            "measured, and do not convert.",
            "<b>Build the case study contractually</b>, in the founding-customer agreement. "
            "Asking afterwards gets a polite no.",
            "<b>Ask for referrals at the ROI review</b>, when the number is in front of "
            "them — not at renewal, when they are thinking about cost.",
            "<b>Write down every lost deal's reason.</b> Ten losses tell you more than your "
            "first win does.",
        ]),
        ("PRICING", AQUA, [
            "<b>Price the outcome.</b> Benchmark against the loaded cost of the person "
            "doing the work today, not against a software subscription.",
            "<b>Protect the margin.</b> Implementation fee covers real delivery cost. If it "
            "does not, the price is wrong.",
            "<b>Keep implementation and recurring separate.</b> They fund different things "
            "and they qualify differently.",
            "<b>Charge for every custom integration.</b> Separately, and quoted after "
            "scoping.",
            "<b>Never promise unlimited usage.</b> There is no hard cost ceiling in the "
            "product. The customer's own AWS account is the correct answer and a good one.",
            "<b>Raise prices after the second customer.</b> The founding price is a "
            "time-limited instrument, and it must be named as one from the first email.",
        ]),
        ("DELIVERY", ACCENT, [
            "<b>Discovery before quoting.</b> Watch the work being done, in person if you "
            "can.",
            "<b>Architecture and security review before deployment.</b> Their IT person or "
            "MSP will be involved eventually — better on your terms in week one.",
            "<b>Deploy with Terraform, never by hand.</b> A hand-built deployment is one you "
            "cannot reproduce, and the second customer is where that bill arrives.",
            "<b>Validate before handover.</b> Health, audit, isolation, permissions, "
            "restart, restore. §19.",
            "<b>Train the humans, not just the software.</b> The approvals queue is a new "
            "habit for somebody. Sit with them.",
            "<b>Monitor from day one.</b> CloudWatch, the audit log, and a weekly look at "
            "the model bill.",
            "<b>Optimise monthly for the first quarter.</b> Personas and Green/Yellow/Red "
            "lines both need tuning once real work flows through them.",
        ]),
    ]
    for title, colour, items in groups:
        body = [Paragraph(f'<font color="#{colour.hexval()[2:]}"><b>{title}</b></font>',
                          S["h3"])]
        body += [Paragraph(f"<bullet>&#8226;</bullet>{i}", S["bullet"]) for i in items]
        s += [Panel(body, accent=colour, fill=PANEL_HI, pad=5 * mm), Gap(3)]
    return s


# ---------------------------------------------------------------- PART 17


GLOSSARY = [
    ("Docker", "A way of packaging software so it runs the same everywhere. A <b>Docker "
     "image</b> is the package; a <b>container</b> is that package running. NOVA ships as "
     "one image, 304 MB."),
    ("ECR", "<b>Elastic Container Registry.</b> Amazon's private warehouse for Docker "
     "images. You upload the NOVA image here; the server downloads it from here."),
    ("EC2", "<b>Elastic Compute Cloud.</b> A computer you rent from Amazon by the hour. "
     "NOVA runs on exactly one."),
    ("EBS", "<b>Elastic Block Store.</b> A hard disk you attach to that computer. It "
     "survives the computer being restarted or replaced — this is where all your state "
     "lives."),
    ("Terraform", "A tool that builds cloud infrastructure from a written description. "
     "You run one command and it creates the server, disk, permissions and logging. It can "
     "also show you what it is <i>about</i> to do before it does it, and undo it all later."),
    ("IAM", "<b>Identity and Access Management.</b> AWS's permission system. An IAM "
     "<b>role</b> is a set of permissions something can wear. NOVA's role is deliberately "
     "tiny."),
    ("Permissions boundary", "A ceiling on an IAM role. Even if someone later attaches a "
     "policy granting more, the boundary still refuses. A seatbelt for permissions."),
    ("SSM", "<b>Systems Manager (Session Manager).</b> Gets you a command line on the "
     "server <i>without</i> opening any port to the internet. This is why NOVA's server has "
     "no inbound access at all."),
    ("Bedrock", "Amazon's service for calling AI models. You are billed per token of text "
     "in and out."),
    ("S3", "<b>Simple Storage Service.</b> Amazon's file storage. Optionally where a "
     "customer's document library lives, mirrored into NOVA's knowledge base."),
    ("CloudWatch", "Where AWS keeps logs and metrics. NOVA's container logs go here."),
    ("Secrets Manager", "Encrypted storage for passwords and API keys. NOVA may read "
     "secrets under one named prefix and nowhere else."),
    ("KMS", "<b>Key Management Service.</b> Holds the encryption key for the disk and the "
     "logs. The customer can supply their own — and for a professional-services firm, "
     "should."),
    ("IMDSv2", "A hardened way for the server to read its own identity. Required here. It "
     "closes a well-known attack where a bug in a web app is used to steal the server's "
     "AWS credentials."),
    ("Image digest", "A fingerprint of an exact image, like "
     "<font face='Courier' size='7'>sha256:fd57…</font>. A tag can be moved to point at "
     "different content; a digest cannot. Always deploy by digest."),
    ("Tenant bundle", "The folder of YAML files describing one customer: their agents, "
     "policy, knowledge sources, channels and automations. NOVA's input."),
    ("Apply", "The command that reads the tenant bundle and writes the runtime "
     "configuration. It <b>overwrites</b> derived files, which is why edits belong in the "
     "bundle and never in the generated output."),
]


STEPS = [
    # (n, title, what, why, expect, wrong, fix, security)
    (1, "Create an AWS account",
     "Sign up at aws.amazon.com. Use the <b>customer's</b> account, not yours.",
     "In the model you are selling, the customer owns the infrastructure and the data. It "
     "is also the answer to their biggest objection.",
     "A root login and a 12-digit account number.",
     "Using your own account 'to make it easier' — you become the data processor and the "
     "whole security story collapses.",
     "Start again in the customer's account. Do not migrate later; it is worse than "
     "starting over.",
     "Turn on MFA for the root user immediately, then never use root again. Create an admin "
     "IAM user for the work."),
    (2, "Choose an AWS region",
     "Pick one region and use it for everything. For a UK customer: <b>eu-west-2 "
     "(London)</b>.",
     "Data residency. A UK accountancy firm's clients' records should sit in the UK, and a "
     "partner will ask. Region also affects which Bedrock models are available and what "
     "they cost.",
     "A region code you will type into Terraform and the AWS console.",
     "Resources scattered across regions — invisible costs and a data-residency answer you "
     "cannot give.",
     "Delete and rebuild in the right region. Terraform makes this cheap, which is one of "
     "the reasons to use it.",
     "Check Bedrock model availability in eu-west-2 <i>before</i> promising a specific "
     "model. Availability differs by region."),
    (3, "Install the local tools",
     "On your own machine: Docker, the AWS CLI v2, Terraform, and the AWS Session Manager "
     "plugin.",
     "These four do everything: build the package, talk to AWS, build the infrastructure, "
     "and get a shell on the server.",
     "Each responds to <font face='Courier' size='7'>--version</font>.",
     "Terraform v1 syntax against an older binary; Session Manager plugin forgotten until "
     "step 18.",
     "Install the plugin now, not when you need it.",
     "Configure the CLI with an IAM user that has admin in this account, not root. Never "
     "commit credentials to a repository."),
    (4, "Build the NOVA Docker image",
     "<font face='Courier' size='7'>deploy/docker/build.sh</font>",
     "The Terraform module pulls an image; it does not build one. This is the package.",
     "A tag of the form <font face='Courier' size='7'>nova-control-plane:0.1.0-g&lt;commit"
     "&gt;</font> and a sha256 digest. About 304 MB.",
     "A dirty working tree, which appends <font face='Courier' size='7'>-dirty.&lt;hash&gt;"
     "</font> to the tag. That is the script working correctly — but do not deploy it.",
     "Commit your work and rebuild. <font face='Courier' size='7'>build.sh --print-tag</font> "
     "tells you the tag for the tree you are on.",
     "The script also writes <font face='Courier' size='7'>:local</font>. That tag is "
     "mutable and must never be deployed."),
    (5, "Test the image locally",
     "<font face='Courier' size='7'>deploy/docker/validate-local.sh &lt;tag&gt;</font>",
     "49 checks: two tenants side by side, authentication, RBAC, isolation, CSP, graceful "
     "shutdown, and that no secret is in the image. Ten minutes here saves a day later.",
     "<b>49 checks, 49 passed, 0 failed.</b>",
     "Anything other than 49/49.",
     "Stop. Do not push. A failure here is a failure in the customer's account too, just "
     "harder to see.",
     "Includes eight checks specifically for secrets in the image. Never skip them because "
     "you are in a hurry."),
    (6, "Create the ECR repository",
     "In the customer's account, in your chosen region, create a private repository named "
     "<font face='Courier' size='7'>nova-control-plane</font>.",
     "Somewhere private to keep the image. The Terraform module does not create it — it "
     "only grants permission to pull from it.",
     "A repository URI like <font face='Courier' size='7'>&lt;account&gt;.dkr.ecr."
     "eu-west-2.amazonaws.com/nova-control-plane</font>.",
     "Creating it as public. Creating it in a different region from the server.",
     "Delete and recreate. Private, same region.",
     "<b>Private.</b> Turn on <b>tag immutability</b> so a tag can never be silently "
     "repointed at different content. Turn on scan-on-push."),
    (7, "Authenticate Docker to ECR",
     "<font face='Courier' size='7'>aws ecr get-login-password --region eu-west-2 | docker "
     "login --username AWS --password-stdin &lt;account&gt;.dkr.ecr.eu-west-2.amazonaws.com"
     "</font>",
     "ECR is private, so Docker needs a token to upload.",
     "<font face='Courier' size='7'>Login Succeeded</font>.",
     "Token expired (they last 12 hours). Wrong region. Wrong AWS profile.",
     "Re-run it. If it still fails, check <font face='Courier' size='7'>aws sts "
     "get-caller-identity</font> — you are probably in the wrong account.",
     "Never paste the password into a command line; the pipe above keeps it out of your "
     "shell history."),
    (8, "Push the image",
     "Tag the local image with the ECR URI and <font face='Courier' size='7'>docker push"
     "</font> it.",
     "Gets the package into the customer's account so the server can download it.",
     "Layers uploading, then a digest printed at the end.",
     "Pushing the <font face='Courier' size='7'>:local</font> tag. Pushing a "
     "<font face='Courier' size='7'>-dirty</font> tag.",
     "Push the immutable tag from step 4. Delete anything wrong from ECR.",
     "This is the first time NOVA code enters the customer's account. Everything after this "
     "runs from what you pushed here."),
    (9, "Get the immutable image digest",
     "From the push output, or: <font face='Courier' size='7'>aws ecr describe-images "
     "--repository-name nova-control-plane --image-ids imageTag=&lt;tag&gt;</font>",
     "A tag is a label that can be moved. A digest is the content itself. Deploying by "
     "digest means the server can only ever run the exact bytes you tested.",
     "<font face='Courier' size='7'>sha256:</font> followed by 64 hex characters.",
     "Using the tag in <font face='Courier' size='7'>image_uri</font> instead. It will "
     "work, and then one day it will run something you did not test.",
     "Use <font face='Courier' size='7'>&lt;repo&gt;@sha256:&lt;digest&gt;</font>.",
     "<b>This is a supply-chain control.</b> It is the difference between a reproducible "
     "deployment and a hopeful one."),
    (10, "Configure the tenant bundle",
     "Copy <font face='Courier' size='7'>nova/examples/acme</font> and edit it for this "
     "customer: organization, identity, agents, policy, knowledge, channels.",
     "This is the customer's workforce, written down. It is also the thing the Control "
     "Center edits — nothing here is a one-off script.",
     "<font face='Courier' size='7'>python -m nova validate &lt;bundle&gt;</font> passes.",
     "A permission not defined in the policy; a knowledge source id typo; an agent "
     "delegating to an agent that does not exist. All caught at load, by design.",
     "Read the error — it names the file and the field.",
     "Never put a credential in the bundle. Six secret-shaped keys are refused at parse "
     "time at any nesting depth, and that refusal is deliberate."),
    (11, "Configure principals",
     "<font face='Courier' size='7'>nova token new &lt;name&gt; --role admin</font> for each "
     "person. Write the printed lines into <font face='Courier' size='7'>control-principals."
     "yaml</font> on the state volume.",
     "Who may use the Control Center, and at what level. Admins change things; viewers "
     "watch.",
     "A token printed <b>once</b>, and a YAML entry containing only its SHA-256 digest.",
     "Losing the token. It is genuinely unrecoverable — NOVA never stores it.",
     "Mint a new one and replace the entry.",
     "<b>The command does not write the file — you do.</b> Without a principals file any "
     "loopback caller is a local admin, and a non-loopback bind is refused outright. Give "
     "staff <b>viewer</b>; partners get admin."),
    (12, "Review the Terraform variables",
     "Copy <font face='Courier' size='7'>terraform.tfvars.example</font> and fill in: "
     "tenant_id, region, vpc_id, subnet_id, image_uri, and optionally bedrock_model_ids, "
     "secret_prefix and integrations.",
     "These are the only decisions the module needs. Everything else has a safe default.",
     "A tfvars file with no placeholder left in it.",
     "Leaving AWS's documented example values (<font face='Courier' size='7'>111122223333"
     "</font>, <font face='Courier' size='7'>vpc-0123…</font>) in place.",
     "Replace every one. Terraform will not catch this — they are syntactically valid.",
     "<b>The subnet needs outbound internet</b> (a NAT gateway, or VPC endpoints). Do not "
     "solve that by putting the server in a public subnet with a public IP."),
    (13, "terraform init",
     "<font face='Courier' size='7'>terraform init</font> in <font face='Courier' size='7'>"
     "deploy/aws</font>.",
     "Downloads the AWS provider and sets up state tracking.",
     "<font face='Courier' size='7'>Terraform has been successfully initialized!</font>",
     "No internet; a provider version conflict.",
     "<font face='Courier' size='7'>terraform init -upgrade</font>.",
     "For anything beyond a first test, configure <b>remote state</b> in S3 with locking. "
     "Local state on a laptop is a single point of failure for the customer's "
     "infrastructure."),
    (14, "terraform validate",
     "<font face='Courier' size='7'>terraform validate</font>",
     "Checks the configuration is internally consistent before it talks to AWS.",
     "<font face='Courier' size='7'>Success! The configuration is valid.</font>",
     "A typo in a variable name.",
     "Fix what it names and re-run.",
     "Free and instant. Always run it."),
    (15, "terraform plan",
     "<font face='Courier' size='7'>terraform plan -out=tfplan</font>",
     "Shows exactly what will be created, changed or destroyed. Nothing happens yet.",
     "A list ending in <font face='Courier' size='7'>Plan: 13 to add, 0 to change, 0 to "
     "destroy</font>.",
     "Credentials for the wrong account. A subnet that is not in the stated VPC.",
     "<font face='Courier' size='7'>aws sts get-caller-identity</font> to confirm which "
     "account you are pointing at.",
     "<b>Never run apply without a saved plan.</b> Applying an unreviewed plan in a "
     "customer's account is the fastest way to lose their trust."),
    (16, "Read the plan properly",
     "Actually read it. All 13 resources.",
     "This is the last moment before anything exists. In a customer's account, it is also "
     "the artefact you can show them.",
     "1 EC2 instance · 1 EBS volume + attachment · 1 KMS key + alias · 1 CloudWatch log "
     "group · 1 security group with <b>egress 443 only and no ingress</b> · 2 IAM roles · 2 "
     "role policies · 1 permissions boundary policy · 1 attachment · 1 instance profile.",
     "Any ingress rule. A larger instance than agreed. A missing KMS key.",
     "Do not apply. Fix the variables.",
     "<b>Confirm there is no ingress rule and that IMDSv2 is required.</b> Those two lines "
     "are most of the host's security posture. Save the plan output as deployment evidence."),
    (17, "terraform apply",
     "<font face='Courier' size='7'>terraform apply tfplan</font>",
     "Builds it.",
     "Two to four minutes, then outputs including the instance id.",
     "Service quota exceeded; a region that does not offer the instance type; insufficient "
     "IAM permissions on your own user.",
     "Read the error — AWS names the quota or the permission. Quota increases are a support "
     "request and can take a day; do this before a customer is watching.",
     "You have now created billable resources in the customer's account. Set up the billing "
     "alarm in §18 <b>today</b>, not next week."),
    (18, "Connect using SSM",
     "<font face='Courier' size='7'>aws ssm start-session --target &lt;instance-id&gt; "
     "--region eu-west-2</font>",
     "Gets you a shell on the server without any open port.",
     "A shell prompt.",
     "<font face='Courier' size='7'>TargetNotConnected</font> — usually the instance has "
     "not finished booting, or the subnet has no route out to reach the SSM service.",
     "Wait two minutes. Then check the subnet's outbound route or its VPC endpoints.",
     "<b>This is the only way in, and that is the design.</b> No SSH key exists. No bastion. "
     "Every session is logged by AWS. Do not 'temporarily' add an SSH rule."),
    (19, "Verify the EC2 host",
     "Check the volume is mounted: <font face='Courier' size='7'>df -h /var/lib/nova</font>",
     "State must be on the encrypted EBS volume, not the instance's own disk.",
     "An XFS filesystem mounted at <font face='Courier' size='7'>/var/lib/nova</font>.",
     "Not mounted — user_data failed part-way.",
     "<font face='Courier' size='7'>sudo cat /var/log/cloud-init-output.log</font> and read "
     "the failure.",
     "If state is on the instance disk instead, everything is lost when the instance is "
     "replaced. Check this properly."),
    (20, "Verify Docker and the container",
     "<font face='Courier' size='7'>sudo docker ps</font>",
     "Confirms the image pulled and the container is running.",
     "One container from your ECR digest, status <font face='Courier' size='7'>healthy"
     "</font> after about 20 seconds.",
     "<font face='Courier' size='7'>ImagePullBackOff</font> or an authentication error — "
     "the instance role could not pull from ECR. A restart loop.",
     "Check the ECR repository policy and that <font face='Courier' size='7'>image_uri</font> "
     "names the right account. For a restart loop: "
     "<font face='Courier' size='7'>sudo docker logs &lt;id&gt;</font> — most often 'no "
     "tenant bundle'.",
     "<b>The image ships no tenant configuration and refuses to serve without one.</b> "
     "Terraform does not place the bundle; you do. Decide how — synced from S3 at boot, or "
     "written during provisioning."),
    (21, "Verify NOVA's health",
     "<font face='Courier' size='7'>sudo docker exec &lt;id&gt; /usr/local/bin/nova-healthcheck"
     "</font>, or curl the health route from inside the container.",
     "Confirms the control plane is accepting connections.",
     "Exit code 0. HTTP 200, 401 or 403 all count as healthy.",
     "Expecting only 200 and concluding it is broken.",
     "Nothing to fix. <b>401 is healthy</b> — with a TLS proxy declared, the control plane "
     "stops trusting loopback, so an unauthenticated probe correctly gets 401. That proves "
     "both the server and its auth layer are working.",
     "This is liveness, not readiness. It answers before a tenant bundle is valid — "
     "deliberately, so a typo in a policy file does not cause a restart loop."),
    (22, "Reach the Control Center securely",
     "Port-forward over SSM: <font face='Courier' size='7'>aws ssm start-session --target "
     "&lt;id&gt; --document-name AWS-StartPortForwardingSession --parameters "
     "'{\"portNumber\":[\"8787\"],\"localPortNumber\":[\"8787\"]}'</font> then open "
     "<font face='Courier' size='7'>http://localhost:8787</font>.",
     "There is no inbound access and the container publishes no port by default. This "
     "tunnels through SSM instead of opening anything.",
     "The Control Center in your browser, on your own machine.",
     "<b>This step needs a decision that has not been made.</b> As shipped, the systemd unit "
     "publishes no ports, so nothing on the host can reach 8787 at all.",
     "Either add <font face='Courier' size='7'>--publish 127.0.0.1:8787:8787</font> to the "
     "unit and set <font face='Courier' size='7'>NOVA_BIND_HOST=0.0.0.0</font> plus "
     "<font face='Courier' size='7'>NOVA_BEHIND_TLS_PROXY=1</font> and a principals file — "
     "or skip the dashboard and use <font face='Courier' size='7'>docker exec</font> for CLI "
     "work.",
     "<b>Never solve this by adding a security-group ingress rule.</b> A non-loopback bind "
     "is refused without a principals file AND a TLS statement — both refusals confirmed "
     "firing. That guard is protecting the customer."),
    (23, "Create the tenant",
     "Place the bundle at <font face='Courier' size='7'>/var/lib/nova/bundle</font> and run "
     "<font face='Courier' size='7'>python -m nova apply</font>.",
     "Turns the declarations into runtime profiles on disk.",
     "Profiles created, an audit log written, and honest warnings about anything not yet "
     "possible.",
     "Warnings that each agent 'cannot run yet' because credentials are missing.",
     "That is correct behaviour, not an error. Step 25 fixes it.",
     "Apply <b>overwrites</b> derived files, including each agent's persona. Edit the "
     "bundle, never the generated output."),
    (24, "Create the agents",
     "Through the Control Center, or by adding YAML to the bundle and applying.",
     "The workforce itself.",
     "Agents visible on the Agents screen, each with a materialised profile.",
     "A permission the policy does not define; an id that is not a valid identifier.",
     "The error names the field. Fix the bundle.",
     "Every agent write is admin-only and audited intent → committed."),
    (25, "Configure the model provider",
     "Set the model in the agent's spec, and put the credential in "
     "<font face='Courier' size='7'>&lt;profile&gt;/.env</font>.",
     "Without a credential an agent is configured and cannot run. NOVA reports this "
     "honestly rather than failing silently.",
     "<font face='Courier' size='7'>nova doctor</font> stops saying 'cannot run yet'.",
     "Exporting the key into the host environment instead of the profile's own file.",
     "Put it in the profile's <font face='Courier' size='7'>.env</font>.",
     "<b>This matters more than it looks.</b> A credential in the host environment is "
     "readable by <i>every</i> agent on that host — verified. Isolation holds only when "
     "credentials live in the per-agent store."),
    (26, "Configure knowledge",
     "Declare corpora in <font face='Courier' size='7'>knowledge.yaml</font>, put documents "
     "in the directory (or upload through the Control Center), and reindex.",
     "What the agents can actually answer from.",
     "Documents listed, an index built, chunk counts shown.",
     "'0 indexed' — usually the include globs do not match the files.",
     "Check the patterns. The Control Center shows what the corpus accepts.",
     "The walk refuses symlinks and anything resolving outside the root. That is the "
     "security boundary of the whole capability — do not work around it."),
    (27, "Configure S3-backed knowledge (optional)",
     "Add an <font face='Courier' size='7'>origin:</font> block naming the bucket, prefix "
     "and region, then Sync.",
     "Lets the firm keep documents where they already are instead of copying them onto the "
     "host by hand.",
     "A sync report: downloaded, unchanged, skipped-and-why.",
     "An access-denied error — the instance role has no S3 permission by default.",
     "Grant <font face='Courier' size='7'>s3:ListBucket</font> and "
     "<font face='Courier' size='7'>s3:GetObject</font> on that one prefix, via the "
     "integrations variable. Nothing broader.",
     "<b>A mirrored corpus is read-only from NOVA.</b> Uploads to it are refused, because "
     "the next sync would delete them. The bucket is where documents are managed."),
    (28, "Configure channels (optional)",
     "Declare the connection, grant it to named agents, and put the credential in the "
     "agent's <font face='Courier' size='7'>.env</font>.",
     "How agents send and receive messages.",
     "The channel listed with its granted agents.",
     "Assuming a platform works because it is in the catalogue.",
     "Connect it and send a real message before telling a customer it is available.",
     "<b>Zero of the 22 platforms are field-validated.</b> Also: a webhook platform needs a "
     "publicly reachable HTTPS endpoint, and this deployment has no inbound access by "
     "design. That is a real decision, not a checkbox."),
    (29, "Create a governed automation",
     "Declare it in the bundle or create it through the Control Center.",
     "Recurring work — the Records Chaser's Monday chase.",
     "The automation listed, with its next run time.",
     "It is listed, the next run time passes, and <b>nothing happens</b>.",
     "That is the known gap. The cron ticker lives inside the runtime's gateway; there is "
     "no standalone daemon. A gateway process must be running.",
     "<b>The Automations screen leads with scheduler liveness for exactly this reason.</b> "
     "Prove a scheduled job actually fires before any customer depends on one."),
    (30, "Run a real AI task",
     "Submit an objective, or send a message on a connected channel, and watch a worker "
     "pick it up.",
     "<b>The single most important step in this list.</b> Until this happens, everything "
     "else is configuration.",
     "A task on the Work board moving through states, a real model response, tokens "
     "recorded on the Usage screen.",
     "No provider configured; the policy denying a tool the agent needs; the model id not "
     "available in this region.",
     "Read the Activity screen — every refusal is logged with the rule that caused it.",
     "<b>This has never been done in a real deployment.</b> Nothing in NOVA has called a "
     "real model provider. Budget real time for this step; expect to find things."),
    (31, "Verify logs",
     "CloudWatch Logs in the console, and the Activity screen in the Control Center.",
     "Two different logs answering two different questions: the container's output, and "
     "what the governance layer decided.",
     "Container logs arriving in CloudWatch; agent and error logs readable per agent in the "
     "Control Center.",
     "Nothing in CloudWatch — the log driver or the IAM permission.",
     "Check the unit's <font face='Courier' size='7'>--log-driver awslogs</font> "
     "configuration and the role policy.",
     "Set a log retention period now. Logs default to never expiring and become a "
     "surprisingly large line on the bill."),
    (32, "Verify the audit trail",
     "<font face='Courier' size='7'>/var/lib/nova/home/nova/audit.jsonl</font>",
     "The record of who did what. This is the artefact that makes the governance claim "
     "true.",
     "Paired records: <font face='Courier' size='7'>intent</font> then "
     "<font face='Courier' size='7'>committed</font> or <font face='Courier' size='7'>failed"
     "</font>, each with the authenticated human and a correlation id.",
     "The actor showing as a service name rather than a person — that means someone is "
     "sharing a token.",
     "One principal per human. That is the entire point.",
     "<b>Show this file to the customer during the pilot.</b> It is the most persuasive "
     "artefact you have, and it is worth more than any slide."),
    (33, "Test restart and recovery",
     "Restart the container. Then stop and start the EC2 instance.",
     "Proves state survives, and that you can recover from a host problem.",
     "The container comes back healthy; agents, automations, provenance and audit history "
     "all intact.",
     "State lost — it was not on the mounted volume.",
     "Go back to step 19.",
     "Also run <font face='Courier' size='7'>nova backup</font> and <b>practise a restore</b>. "
     "A backup you have never restored is not a backup. Note that backups exclude "
     "credentials by default, deliberately."),
    (34, "Test tenant isolation",
     "Confirm this deployment holds exactly one tenant and its state is under one root.",
     "One tenant per deployment is the isolation guarantee. There is no multi-tenancy.",
     "One bundle, one tenant id stamped on every audit event.",
     "Attempting to serve a second customer from the same host.",
     "Do not. Deploy a second stack. It is one Terraform apply.",
     "<b>Never co-locate two customers.</b> The architecture does not support it and the "
     "repository says so plainly."),
    (35, "Test permissions",
     "Create a viewer principal. Confirm it can read the agents list and is refused on "
     "policy, decisions, budget and every write. Then attempt a denied tool call and watch "
     "the refusal appear in Activity.",
     "Proves RBAC and the policy layer are live in <i>this</i> deployment, not just in "
     "tests.",
     "403s where expected, and a logged refusal naming the rule.",
     "A viewer seeing something they should not.",
     "Stop and investigate before go-live. Do not proceed.",
     "<b>Do this in front of the customer.</b> A demonstrated refusal is worth more than "
     "any security page on your website."),
    (36, "Record the deployment evidence",
     "Save: the plan output, the image digest, the Terraform outputs, the 49-check result, "
     "screenshots of steps 30–35, and the AWS account and region.",
     "This becomes the repeatable runbook for customer two, and the evidence pack for the "
     "first security questionnaire you are asked to complete.",
     "A dated folder per customer.",
     "Not doing it, then rebuilding the knowledge from memory for the next customer.",
     "Do it while it is fresh — the same day.",
     "Store it where the customer's credentials are not. Evidence packs get emailed."),
]


def part_17():
    s = section("17", "Deploying NOVA to AWS", "Step by step, for a non-engineer")

    s += [P(
        "This assumes you understand business but not cloud infrastructure. Every technical "
        "term is explained the first time it appears. Work through the steps in order — "
        "several of them depend on the one before.", "lede")]

    s += [h2("17.1  What you are actually doing")]
    s += [P(
        "You are putting a software package onto a computer you rent from Amazon, inside "
        "the customer's own Amazon account, with a disk that survives restarts, permissions "
        "that are deliberately tiny, no way in from the internet, and a record of everything "
        "that happens. That is the whole thing.")]
    s += [Flow([
        ("Package it", "Docker builds one image"),
        ("Store it", "ECR, private, in their account"),
        ("Build the infra", "Terraform creates 13 resources"),
        ("Run it", "EC2 pulls and runs the image"),
        ("Configure", "Tenant bundle, agents, knowledge"),
        ("Verify", "Health, audit, isolation, restart"),
    ], per_row=3)]

    s += [h2("17.2  Every term, explained once")]
    rows = [["Term", "What it actually is"]] + [[t, d] for t, d in GLOSSARY]
    s += [table(rows, [34 * mm, W - 34 * mm], font_size=7.6)]

    s += [h2("17.3  The 36 steps")]
    s += [callout(
        "Before you start",
        "Steps 1–17 build infrastructure. Steps 18–29 configure NOVA. Steps 30–36 prove it "
        "works. <b>Step 30 — running a real AI task — has never been performed in any "
        "deployment.</b> Budget real time for it and expect to find problems; that is what "
        "first deployments are for.", "warn")]

    for n, title, what, why, expect, wrong, fix, sec in STEPS:
        body = [
            Paragraph(
                f'<font color="#{ACCENT.hexval()[2:]}" size="13"><b>{n:02d}</b></font>'
                f'&nbsp;&nbsp;<b>{title}</b>', S["h3"]),
            Paragraph(f"<b>What you're doing.</b> {what}", S["cell"]),
            Gap(1.6),
            Paragraph(f"<b>Why.</b> {why}", S["cell"]),
            Gap(1.6),
            Paragraph(f'<font color="#{GOOD.hexval()[2:]}"><b>You should see.</b></font> '
                      f'{expect}', S["cell"]),
            Gap(1.6),
            Paragraph(f'<font color="#{WARNING.hexval()[2:]}"><b>What can go wrong.</b>'
                      f'</font> {wrong}', S["cell"]),
            Gap(1.6),
            Paragraph(f'<font color="#{AQUA.hexval()[2:]}"><b>How to fix it.</b></font> '
                      f'{fix}', S["cell"]),
            Gap(1.6),
            Paragraph(f'<font color="#{CRITICAL.hexval()[2:]}"><b>Security.</b></font> '
                      f'{sec}', S["cell"]),
        ]
        s.append(Panel(body, fill=PANEL_HI, accent=ACCENT, pad=4.4 * mm))
        s.append(Gap(3))
    return s


# ---------------------------------------------------------------- PART 18


def part_18():
    s = section("18", "AWS cost control", "Structure, not prices")

    s += [callout(
        "Why there are no prices in this section",
        "AWS pricing changes, varies by region, and is only authoritative on AWS's own "
        "pages — which were <b>unreachable from the environment this document was built "
        "in</b> (the network egress policy blocks aws.amazon.com). Quoting figures from "
        "memory or from a third-party blog would be exactly the kind of fabricated number "
        "this document exists to avoid. What follows is the cost <i>structure</i>, which "
        "does not change, plus the official URLs to price it against on the day you quote.",
        "warn")]

    s += [h2("18.1  What generates a bill")]
    s += [table([
        ["Service", "What you pay for", "How big it is, relatively", "How to control it"],
        ["<b>EC2</b>", "The server, per hour it exists — running or not, unless stopped",
         "<b>Largest fixed cost.</b> A general-purpose instance running continuously.",
         "Right-size it. Start small; NOVA's control plane is not compute-heavy. Consider a "
         "Savings Plan once the size is settled."],
        ["<b>EBS</b>", "The disk, per GB per month, whether used or not",
         "Small but permanent — and it survives the instance.",
         "Provision what you need. Snapshots also cost; set a retention policy."],
        ["<b>ECR</b>", "Image storage per GB/month, plus data transfer",
         "Very small. The image is ~304 MB.",
         "A lifecycle policy to expire untagged images. Otherwise every build accumulates."],
        ["<b>CloudWatch Logs</b>", "Ingestion per GB, then storage per GB/month",
         "<b>The classic surprise.</b> Logs default to never expiring.",
         "<b>Set a retention period on day one.</b> 30 or 90 days. This is the single "
         "highest-value cost action in this table."],
        ["<b>S3</b>", "Storage, requests, and retrieval",
         "Small for a document library; the request count can surprise you.",
         "Lifecycle rules. Note that NOVA's sync compares ETags and does not re-download "
         "unchanged objects, which keeps request volume down."],
        ["<b>Bedrock</b>", "Per 1,000 tokens in and out, per model",
         "<b>The only variable that can grow without limit.</b> Depends entirely on agent "
         "activity and model choice.",
         "See §18.3. This is the one to watch weekly."],
        ["<b>Data transfer</b>", "Mostly outbound to the internet; NAT gateway processing",
         "Modest here — no public traffic — but a NAT gateway has its own hourly charge "
         "plus per-GB.",
         "VPC endpoints for ECR, S3, SSM and CloudWatch instead of a NAT gateway can be "
         "cheaper. Price both."],
        ["<b>KMS</b>", "Per key per month, plus API requests",
         "Small and fixed.", "One key. A customer-managed key is worth the small cost for "
         "this buyer."],
        ["<b>Secrets Manager</b>", "Per secret per month, plus API calls",
         "Small.", "Few secrets. Do not put one secret per agent per variable in here "
         "unless you need to."],
    ], [26 * mm, 40 * mm, 44 * mm, W - 110 * mm], font_size=7.4)]

    s += [h2("18.2  Price it here, on the day")]
    s += [Panel([
        Paragraph("Official AWS pricing pages — the only authoritative source", S["h3"]),
        Paragraph("EC2 on-demand: <font face='Courier' size='7.4'>"
                  "https://aws.amazon.com/ec2/pricing/on-demand/</font>", S["cell"]),
        Paragraph("EBS: <font face='Courier' size='7.4'>https://aws.amazon.com/ebs/pricing/"
                  "</font>", S["cell"]),
        Paragraph("ECR: <font face='Courier' size='7.4'>https://aws.amazon.com/ecr/pricing/"
                  "</font>", S["cell"]),
        Paragraph("CloudWatch: <font face='Courier' size='7.4'>"
                  "https://aws.amazon.com/cloudwatch/pricing/</font>", S["cell"]),
        Paragraph("S3: <font face='Courier' size='7.4'>https://aws.amazon.com/s3/pricing/"
                  "</font>", S["cell"]),
        Paragraph("Bedrock: <font face='Courier' size='7.4'>"
                  "https://aws.amazon.com/bedrock/pricing/</font>", S["cell"]),
        Paragraph("KMS: <font face='Courier' size='7.4'>https://aws.amazon.com/kms/pricing/"
                  "</font>", S["cell"]),
        Paragraph("Secrets Manager: <font face='Courier' size='7.4'>"
                  "https://aws.amazon.com/secrets-manager/pricing/</font>", S["cell"]),
        Paragraph("Data transfer: <font face='Courier' size='7.4'>"
                  "https://aws.amazon.com/ec2/pricing/on-demand/#Data_Transfer</font>",
                  S["cell"]),
        Gap(2),
        Paragraph("Build the estimate in the <b>AWS Pricing Calculator</b> "
                  "(<font face='Courier' size='7.4'>https://calculator.aws</font>), save "
                  "it, and send the customer the link. It is a shareable URL, it is "
                  "authoritative, and it moves the cost conversation from your word to "
                  "Amazon's.", S["cell"]),
    ], fill=PANEL_HI, accent=ACCENT)]

    s += [Gap(3), callout(
        "“It runs on AWS” does not mean it is free — say so early",
        "A non-technical buyer often hears “your own AWS account” as “no extra cost”. "
        "Correct that in the first conversation, not in month two when the bill arrives. "
        "The honest framing: <b>“You pay Amazon directly for the server and the AI usage. "
        "That's typically a few hundred pounds a month at this size, it's in your account "
        "so you can see every line of it, and I'll help you set a budget alarm on day one.”"
        "</b> Then actually help them set it.", "warn")]

    s += [h2("18.3  Beginner checklist for avoiding a surprise bill")]
    s += [Checklist([
        "Billing alerts on, with an email that someone reads",
        "AWS Budgets: a monthly budget with an 80% forecast alert",
        "CloudWatch log retention set (30 or 90 days) — not 'never expire'",
        "ECR lifecycle policy to expire untagged images",
        "S3 lifecycle rules if a document library is large",
        "Instance right-sized — start small, measure, then resize",
        "Nothing left running from testing (old instances, orphaned volumes, old snapshots)",
        "Orphaned EBS volumes deleted — they bill after the instance is gone",
        "NAT gateway vs VPC endpoints priced both ways before choosing",
        "Bedrock usage reviewed weekly for the first month",
        "Cost allocation tags on every resource, so the customer can see NOVA's share",
        "A named person on the customer's side who sees the bill",
    ], columns=2)]

    s += [Gap(4), callout(
        "There is no hard spending ceiling in NOVA — plan around it",
        "This is stated in §3.8 and it bears repeating here because it is a money question. "
        "Run-budget seconds trigger a soft wrap-up message; nothing terminates. Token and "
        "cost figures on the Usage screen are <b>observed, not enforced</b>. The controls "
        "that actually exist are: AWS Budgets alarms, the per-run tool-call ceiling in the "
        "compiled policy, a conservative model choice, and looking at the bill weekly. "
        "<b>Never tell a customer NOVA caps their spend.</b>", "risk")]
    return s


# ---------------------------------------------------------------- PART 19


def part_19():
    s = section("19", "Production checklist", "Two gates, before and during")

    s += [h2("19.1  Before any customer")]
    s += [P("These are about <i>you</i>, not about a deployment. Do not take a first "
            "customer's money with more than two of these unticked.")]
    s += [Panel([Checklist([
        "Security review of the deployment completed and written down",
        "Tenant isolation verified on real infrastructure",
        "IAM policies reviewed line by line by someone other than the author",
        "TLS decision made and documented (proxy, or certificate)",
        "Authentication proven: principals file, hashed tokens, no shared logins",
        "RBAC proven: viewer refused on every admin route",
        "Secrets: none in the image, none in the bundle, none in the audit log",
        "Backup taken AND a restore practised on a fresh host",
        "Logging: CloudWatch delivery confirmed, retention set",
        "Monitoring: health check alarm, disk-space alarm, billing alarm",
        "Cost controls: budget, alerts, log retention, ECR lifecycle",
        "Recovery: instance replacement tested with state intact",
        "Upgrade procedure written — new digest, re-apply, verify",
        "Rollback procedure written and tested — previous digest, redeploy",
        "Deployment runbook written from a real deployment, not from this document",
        "Capability schedule (§3) current, and matching every sales claim you make",
    ], columns=2)], fill=PANEL_HI, accent=ACCENT)]

    s += [Gap(4), h2("19.2  Customer deployment")]
    s += [Panel([Checklist([
        "Discovery: workflow observed, baseline measurement started",
        "Architecture agreed with the customer's IT person or MSP",
        "AWS account access confirmed, region agreed (eu-west-2 for UK)",
        "IAM: admin user created, root MFA on, root not used again",
        "ECR repository created, private, tag-immutable",
        "Image pushed; deployment pinned to the digest, not a tag",
        "Terraform plan reviewed WITH the customer and saved as evidence",
        "Terraform applied; 13 resources confirmed; no ingress rule",
        "Tenant bundle placed and validated",
        "Principals file written; partners admin, staff viewer",
        "Model provider configured; credentials in the profile .env, not the host",
        "Knowledge corpora loaded and indexed; counts verified",
        "Integrations: S3 origin and/or channels, each granted per agent",
        "Agents created; personas written with the customer",
        "Policy compiled; a denied action demonstrated to the customer",
        "Green/Yellow/Red agreed, signed, and reflected in the policy",
        "Automations declared; scheduler liveness confirmed firing",
        "A real AI task executed end to end and watched",
        "Validation: health, logs, audit trail, restart, isolation, permissions",
        "Training delivered — especially the approvals queue habit",
        "Monitoring and budget alarms live before go-live, not after",
        "Deployment evidence pack saved and dated",
        "Day-60 ROI review booked in the calendar now",
    ], columns=2)], fill=PANEL_HI, accent=GOOD)]
    return s


# ---------------------------------------------------------------- PART 20


def part_20():
    s = section("20", "12-month commercial roadmap", "Ordered by customer value")

    s += [P(
        "Nothing here is a feature somebody imagined. Every product item either closes a "
        "gap identified in §3 and §23, or was asked for by a real customer — and in months "
        "1–3 there are no customers, so everything is gap-closing. That is the correct "
        "order.", "lede")]

    s += [Timeline([
        ("MONTH 1", "Field validation — nothing else matters",
         "Deploy to a real AWS account. Push to ECR. Prove ECR pull, KMS volume, awslogs, "
         "SSM, IMDSv2. Execute one real model call through Bedrock. Prove one scheduled job "
         "fires. Settle the Control Center exposure decision.", CRITICAL),
        ("MONTH 2", "Make it repeatable, and start the list",
         "Second deployment from the runbook with no improvisation. Backup and restore "
         "drill. Build the 100-prospect list. Record the 90-second demo. Governance page "
         "live.", WARM),
        ("MONTH 3", "First pilot",
         "Outreach at volume. 8–12 discovery calls. First paid pilot signed and started "
         "with a two-week baseline. First security questionnaire answered — keep the "
         "answers.", ACCENT),
        ("MONTH 4", "Deliver, and measure honestly",
         "Pilot weeks 2–6. Weekly reviews. Sample and grade 30 outputs a week. Fix what "
         "the first real workload breaks — it will break something.", ACCENT),
        ("MONTH 5", "Convert and prove",
         "Day-60 ROI review. Convert to standard pricing. Write the case study with their "
         "numbers and their quote. Ask for two referrals in that meeting.", GOOD),
        ("MONTH 6", "Second customer",
         "Sell with the case study. Raise price to standard tier. Onboard the sales partner "
         "(§15) with the capability schedule attached.", GOOD),
        ("MONTH 7", "Customer success becomes a function",
         "Monthly reviews for both customers. First upsell — a second workflow. Task "
         "detail and run history surfaced, because customers will ask.", ACCENT),
        ("MONTH 8", "Security posture",
         "Formal security questionnaire pack. DPIA template for customers. Penetration "
         "test. Start the SOC 2 conversation if mid-market is the target.", WARM),
        ("MONTH 9", "Third and fourth customers",
         "Repeatable sale. Partner produces their first independently-sourced deal. "
         "Vertical-specific content compounding.", GOOD),
        ("MONTH 10", "Vertical specialisation",
         "A pre-built accountancy tenant bundle: standard agents, standard policy, standard "
         "Green/Yellow/Red. Deployment time falls from weeks to days.", ACCENT),
        ("MONTH 11", "Channel",
         "Two or three referral partners: practice-software consultants, accountancy-focused "
         "MSPs, fractional FDs. Their relationship, your product.", AQUA),
        ("MONTH 12", "Assess honestly",
         "Five or six customers, a repeatable deployment, a repeatable sale, one vertical "
         "understood. Decide: hire a salesperson (§15.2), raise, or stay deliberately "
         "small.", GOOD),
    ])]

    s += [PageBreak(), h2("20.1  By function")]
    s += [table([
        ["Function", "Months 1–3", "Months 4–6", "Months 7–9", "Months 10–12"],
        ["<b>Product</b>", "Close the field-validation gaps. No new features.",
         "Fix what the first real workload exposes.",
         "Task detail, run history, comments — customers ask for these.",
         "Accountancy bundle template. Semantic retrieval only if customers actually hit "
         "the BM25 limit."],
        ["<b>Engineering</b>", "SQLite 3.53 build stage. Exposure decision. Gateway process "
         "for the scheduler.",
         "Repeatable deployment. Backup/restore drill. Upgrade and rollback.",
         "Observability. Alerting. On-call basics.",
         "Hardening for mid-market. Multi-deployment management tooling."],
        ["<b>Sales</b>", "100-prospect list. Demo. First 8–12 calls.",
         "First pilot. First conversion. Case study.",
         "Partner onboarded. Customers 3 and 4.",
         "Repeatable process documented. Decide on the first hire."],
        ["<b>Marketing</b>", "Governance page. LinkedIn founder cadence. Demo video.",
         "Case study. ROI method published.",
         "Vertical content. Conference talk or webinar.",
         "Retargeting test. First paid experiment, small."],
        ["<b>Customer success</b>", "n/a", "Weekly pilot reviews. Output sampling.",
         "Monthly reviews. First upsell. Referral asks.",
         "Health scoring. Renewal process. QBRs."],
        ["<b>Security</b>", "Internal review. IAM audit.",
         "First questionnaire. DPIA template.",
         "Penetration test. Incident response plan.",
         "SOC 2 readiness if mid-market is the target."],
        ["<b>Hiring</b>", "Nobody.", "Nobody.", "Commission-only sales partner.",
         "Assess against §15.2. Probably a delivery engineer before a salesperson."],
    ], [24 * mm, (W - 24 * mm) / 4, (W - 24 * mm) / 4, (W - 24 * mm) / 4,
        (W - 24 * mm) / 4], font_size=7)]
    return s


# ---------------------------------------------------------------- PART 21


def part_21():
    s = section("21", "Revenue scenarios", "Illustrative arithmetic, not forecasts")

    s += [callout(
        "Illustrative scenario, not a financial forecast",
        "NOVA has <b>no customers and no revenue</b>. Every figure in this section is "
        "arithmetic on a stated assumption. Its purpose is to show the shape of the "
        "business and where the margin sits — not to predict anything. Do not put these in "
        "an investor deck as projections.", "warn")]

    s += [h2("21.1  The assumptions, stated once")]
    s += [table([
        ["Assumption", "Value used", "Basis"],
        ["Blended implementation fee", "£6,000",
         "Between the founding (£4,500) and standard (£7,500) tiers — §7"],
        ["Blended monthly fee", "£1,200", "Between founding (£950) and standard (£1,450)"],
        ["Your infrastructure cost per customer", "£0",
         "The customer pays AWS directly, in their own account. This is the BYOC model and "
         "it is a real structural advantage"],
        ["Your delivery cost per customer per month", "£150",
         "Assumed: support time, monitoring, your own tooling amortised"],
        ["Churn", "0%", "<b>Assumed away for simplicity. This is unrealistic</b> and the "
         "biggest weakness in these numbers"],
        ["Implementation gross margin", "~50%",
         "Assumed: the rest is your delivery time at a notional rate"],
    ], [50 * mm, 26 * mm, W - 76 * mm], font_size=7.8)]

    s += [h2("21.2  Scenarios")]
    rows = [["Customers", "Setup revenue (one-off)", "MRR", "ARR",
             "Your cost/mo", "Gross margin on recurring"]]
    for n in (5, 10, 25, 50, 100):
        setup = n * 6000
        mrr = n * 1200
        cost = n * 150
        rows.append([f"<b>{n}</b>", f"£{setup:,}", f"£{mrr:,}", f"£{mrr * 12:,}",
                     f"£{cost:,}", f"£{mrr - cost:,}/mo  (<b>87.5%</b>)"])
    s += [table(rows, [20 * mm, 32 * mm, 26 * mm, 30 * mm, 26 * mm, W - 134 * mm],
                font_size=7.8, align={i: "RIGHT" for i in range(1, 6)})]
    s += [caption("ILLUSTRATIVE SCENARIO — not a forecast. Assumes £6,000 setup, "
                  "£1,200/month, £150/month delivery cost per customer, zero churn, and "
                  "that the customer pays AWS directly.")]

    s += [Gap(3), img("revenue-ramp.png")]
    s += [caption("ILLUSTRATIVE SCENARIO — not a forecast. One stated assumption: "
                  "£1,400/month per customer and no churn, with customers landing on the "
                  "ramp shown. Reaching 13 customers in 12 months would be a strong first "
                  "year and is not a prediction.")]

    s += [h2("21.3  What these numbers hide")]
    s += [table([
        ["The number looks like", "The reality"],
        ["87.5% gross margin on recurring",
         "True only if delivery stays at £150/customer/month. The first three customers "
         "will cost far more than that in your own time, and that time is the real "
         "constraint on growth."],
        ["Setup revenue scales linearly",
         "Each implementation is real work. Without the month-10 vertical bundle, "
         "customer 25 costs roughly what customer 5 did — and you only have so many weeks."],
        ["Zero churn", "The single most unrealistic assumption here. Model 10–20% annual "
         "churn for an SMB product with no switching cost and see what the curve does."],
        ["Customers pay AWS, so infrastructure is free to you",
         "True, and a genuine structural advantage — but it also means a customer's model "
         "bill is a churn risk you do not control and cannot cap (§18.3)."],
        ["100 customers is a business",
         "100 single-tenant deployments is 100 things to upgrade, monitor and support. "
         "Without the month-12 deployment tooling, that is an operations problem long "
         "before it is a revenue achievement."],
    ], [46 * mm, W - 46 * mm], font_size=7.8)]
    return s


# ---------------------------------------------------------------- PART 22


def part_22():
    s = section("22", "Competitive positioning", "Where NOVA actually sits")

    s += [img("positioning.png")]
    s += [caption("Two axes a buyer in this market actually chooses on. Positions are my "
                  "assessment of each category, not a scored benchmark of named products.")]

    s += [h2("22.1  Category by category")]
    s += [table([
        ["Category", "Examples", "What they do better", "Where NOVA wins"],
        ["<b>Workflow automation</b>", "Zapier, Make, Power Automate",
         "Vast connector libraries, instant setup, trivially cheap, enormous communities.",
         "They move data between apps on fixed rules. They do not reason over documents, "
         "do not have a policy layer, do not have an approvals queue, and run in someone "
         "else's cloud. Different product for a different job — say so rather than "
         "competing."],
        ["<b>AI copilots</b>", "Microsoft 365 Copilot, Salesforce Agentforce, Gemini for "
         "Workspace",
         "Already in the customer's stack, per-seat pricing, no deployment, huge trust "
         "advantage from the brand.",
         "Confined to their own vendor's data and surfaces. Assist a person rather than "
         "hold a workflow. <b>This is your most common real competitor</b> — the answer is "
         "“Copilot helps your people work faster; NOVA does the job.”"],
        ["<b>Hosted AI agent platforms</b>", "Various 2026 agent-building platforms",
         "Fast to start, growing fast, good developer experience, no infrastructure work.",
         "The customer's data goes to their cloud. Governance is usually developer-facing "
         "rather than business-facing. For a firm holding client confidential records, "
         "“in your own account” is the whole conversation."],
        ["<b>AI automation agencies</b>", "The many UK agencies in §7.1",
         "Cheap to start, flexible, sell on outcomes, move fast.",
         "They build bespoke workflows on someone else's platform, usually with no "
         "governance layer and no audit trail. When the agency goes quiet, the customer has "
         "an unmaintainable thing. <b>This is who you are competing with on price</b>, and "
         "governance is the differentiator."],
        ["<b>Enterprise AI platforms</b>", "watsonx Orchestrate, UiPath, Appian, and similar",
         "Real governance, self-hosted options, compliance certifications, established "
         "sales machinery.",
         "Priced and scoped for a 5,000-person organisation. A 40-person practice cannot "
         "buy them, cannot implement them, and is not worth their sales cycle. <b>That gap "
         "is your market.</b>"],
    ], [30 * mm, 30 * mm, (W - 60 * mm) * 0.44, (W - 60 * mm) * 0.56], font_size=7.2)]

    s += [Gap(3), callout(
        "On Palantir comparisons — don't",
        "It will be tempting, because the shape rhymes: governed, deployed into the "
        "customer's environment, audit-heavy. <b>NOVA is not comparable to Palantir</b> and "
        "claiming it is will end a credible conversation instantly. Different scale, "
        "different problem, different maturity, thousands of engineer-years apart. If "
        "someone raises it, the honest answer is: “Same instinct about governance, "
        "completely different scale. We do one thing for firms of 20 to 150 people.” That "
        "answer earns respect; the alternative does not.", "risk")]

    s += [h2("22.2  What is actually differentiated, and how defensible it is")]
    s += [table([
        ["Differentiator", "How real", "How defensible"],
        ["<b>Runs in the customer's own AWS account</b>", "Real — the Terraform module "
         "exists and is tested. Unproven on real AWS.",
         "<b>Strong.</b> Most competitors are structurally hosted and cannot follow without "
         "rebuilding their business model."],
        ["<b>Governed AI workforce</b> — policy, approvals, per-agent permissions",
         "Real and enforced. Fail-closed policy, compiled per agent, checked at every tool "
         "call.",
         "<b>Strong for 12–24 months.</b> It is genuinely hard to retrofit, and the "
         "hard-won lessons (a plugin that was installed and never consulted) are not "
         "obvious from outside."],
        ["<b>Write-ahead audit trail</b>", "Real and live-proven locally. Intent before, "
         "outcome after, authenticated human.",
         "<b>Moderate.</b> Not hard to build; it is hard to remember to build, and harder "
         "to retrofit into a product that did not start with it."],
        ["<b>Configurable workforce, not code</b>", "Real — agents, personas, permissions "
         "and schedules are all editable in the browser and all write to the bundle.",
         "<b>Moderate.</b> Everyone is building this. Your version's advantage is that "
         "edits are durable through apply, which is less obvious than it sounds."],
        ["<b>Enterprise Control Center</b>", "Real. 13 screens, strict CSP, RBAC, tenant-"
         "scoped API.",
         "<b>Weak alone.</b> UI is copyable. It matters as the delivery vehicle for the "
         "governance, not on its own."],
        ["<b>Built on an open runtime</b>", "Real and honest.",
         "<b>Double-edged.</b> You inherit a large capable runtime for free — and a "
         "dependency you do not control. See §23."],
    ], [44 * mm, (W - 44 * mm) * 0.44, (W - 44 * mm) * 0.56], font_size=7.4)]
    return s


# ---------------------------------------------------------------- PART 23


def part_23():
    s = section("23", "Risks and honest limitations", "Everything that could stop this")

    s += [P(
        "This section is mandatory reading before any sales conversation. Nothing is hidden "
        "and nothing is softened. The last column is the one that matters: whether it blocks "
        "the first customer.", "lede")]

    def risk_rows(rows):
        head = [["Limitation", "Severity", "Impact", "Action", "Blocks #1?"]]
        return table(head + rows, [44 * mm, 20 * mm, (W - 108 * mm) * 0.5,
                                   (W - 108 * mm) * 0.5, 24 * mm], font_size=7.2)

    def sev(level):
        colour = {"CRITICAL": CRITICAL, "HIGH": SERIOUS, "MEDIUM": WARNING,
                  "LOW": TEXT_MUTED}[level]
        return Paragraph(f'<font color="#{colour.hexval()[2:]}"><b>{level}</b></font>',
                         S["cell"])

    def blocks(yes):
        if yes == "YES":
            return Paragraph(f'<font color="#{CRITICAL.hexval()[2:]}"><b>YES</b></font>',
                             S["cell"])
        if yes == "NO":
            return Paragraph(f'<font color="#{GOOD.hexval()[2:]}"><b>No</b></font>',
                             S["cell"])
        return Paragraph(f'<font color="#{WARNING.hexval()[2:]}"><b>{yes}</b></font>',
                         S["cell"])

    s += [h2("23.1  Technical and validation risks")]
    s += [risk_rows([
        ["<b>No AWS field validation.</b> ECR pull, instance profile, KMS volume mount, "
         "awslogs delivery, SSM access and ExternalId integration roles are configured and "
         "have never run.", sev("CRITICAL"),
         "First deployment becomes debugging in front of a paying customer. Any of the six "
         "could fail.",
         "Deploy to your own AWS account this month. §24 action 1.", blocks("YES")],
        ["<b>No real provider validation.</b> No agent has called Bedrock or any hosted "
         "model. The live worker run used a scripted server.", sev("CRITICAL"),
         "Token accounting, cost estimation, streaming, tool-call accumulation and error "
         "handling are all unproven against a real provider.",
         "Execute one real Bedrock call end to end. §24 action 2.", blocks("YES")],
        ["<b>Scheduled execution never observed firing.</b> The cron ticker lives inside "
         "the runtime's gateway; there is no standalone daemon.", sev("CRITICAL"),
         "A deployment can hold a perfectly correct schedule that nothing runs. The Records "
         "Chaser depends on this entirely.",
         "Run a gateway process and watch a job fire. Until then, scope pilots to the "
         "Librarian and Client Desk. §24 action 3.", blocks("YES")],
        ["<b>Control Center exposure undecided.</b> No ingress, no published port, SSM gives "
         "a host shell — which cannot reach port 8787.", sev("CRITICAL"),
         "As shipped, the customer cannot open the dashboard at all.",
         "Choose: loopback publish + SSM port-forward (recommended), or CLI-only. Document "
         "it. §24 action 4.", blocks("YES")],
        ["<b>Zero channels field-validated.</b> 5 read in source, 17 manifest-only.",
         sev("HIGH"),
         "Any channel promised to a customer might not work. Email is the one the first "
         "offer depends on.",
         "Field-validate email first, then one chat platform. Never put a logo on a slide "
         "before connecting it.", blocks("Partly")],
        ["<b>SQLite 3.40.1 in the image</b> carries the upstream WAL-reset bug. The runtime "
         "detects it and degrades to journal_mode=DELETE.", sev("MEDIUM"),
         "No corruption risk — the degradation is safe. What is lost is reader/writer "
         "concurrency, which matters for a busy 24/7 board.",
         "Port the existing SQLite 3.53 builder stage from the root Dockerfile. Blocked in "
         "this environment by egress policy, not by difficulty.", blocks("NO")],
        ["<b>Single-node architecture.</b> State is SQLite on one attached volume.",
         sev("MEDIUM"),
         "No horizontal scale, no HA. An instance failure is downtime until it is replaced.",
         "Correct for this market. Be explicit in the SLA; do not promise HA.", blocks("NO")],
        ["<b>No multi-tenancy.</b> One tenant per deployment.", sev("MEDIUM"),
         "Every customer is a separate stack to deploy, upgrade and monitor. This is an "
         "operations cost that grows linearly.",
         "Accept it — it is also the security story. Build multi-deployment tooling around "
         "month 12, not multi-tenancy.", blocks("NO")],
        ["<b>Backup/restore never drilled on real infrastructure.</b>", sev("HIGH"),
         "A backup you have never restored is not a backup.",
         "Drill it on a fresh host in month 2, before customer one goes live.", blocks("Partly")],
        ["<b>Thin observability.</b> Health check is liveness only; no alerting beyond what "
         "CloudWatch gives you.", sev("MEDIUM"),
         "You will find out a customer's deployment is broken when they tell you.",
         "Health alarm, disk alarm, billing alarm on day one of the first deployment.",
         blocks("NO")],
        ["<b>Positive tool scoping not enforced.</b> toolsets/allow are recorded, not "
         "enforced by the adapter.", sev("MEDIUM"),
         "A narrower grant than the customer believes. Denials ARE enforced, so the "
         "practical exposure is bounded.",
         "Use tools.deny for anything that matters. Never claim allow-listing as an "
         "enforced control.", blocks("NO")],
        ["<b>BM25 keyword retrieval, no embeddings.</b>", sev("LOW"),
         "Poor recall where the customer's question uses different vocabulary from their "
         "documents.",
         "Sell it accurately. Revisit only if a real customer hits the limit.", blocks("NO")],
    ])]

    s += [h2("23.2  Commercial and operational risks")]
    s += [risk_rows([
        ["<b>No hard cost ceiling.</b> Structurally unavailable — run-budget seconds are a "
         "soft wrap-up, not a kill.", sev("HIGH"),
         "A runaway loop on an expensive model produces a bill nobody capped. Also a "
         "renewal risk you do not control.",
         "Customer pays AWS directly; AWS Budgets alarms; conservative model choice; "
         "per-run tool-call ceiling; weekly review. <b>Never claim a cap.</b>", blocks("NO")],
        ["<b>Model costs are variable and the customer sees them.</b>", sev("MEDIUM"),
         "Bill shock in month two kills the relationship faster than a product fault.",
         "Set expectations in the first conversation. Set the alarm on day one. Review "
         "weekly for the first month.", blocks("NO")],
        ["<b>Integration complexity.</b> Every firm's practice software is configured "
         "differently.", sev("MEDIUM"),
         "Implementation overruns eat the fee and your calendar.",
         "Scope one workflow. Quote integrations separately. Say no to bespoke in the first "
         "three customers.", blocks("NO")],
        ["<b>Customer security reviews.</b> No SOC 2, no ISO 27001, no penetration test, no "
         "production history.", sev("HIGH"),
         "Blocks mid-market and kills regulated sectors outright.",
         "Target SMB first, where the partner is the reviewer. Build the questionnaire pack "
         "from the first one you answer.", blocks("NO")],
        ["<b>UK GDPR and ICO expectations.</b> The ICO's January 2026 agentic-AI report "
         "flags purposes set too broadly and unfettered access to data and systems; most "
         "agentic deployments touching personal data warrant a DPIA.", sev("HIGH"),
         "Client records are personal data. A firm that has not done a DPIA is exposed, and "
         "so are you.",
         "<b>Turn this into a sales asset.</b> Provide a DPIA template. Narrow purposes and "
         "per-agent grants are literally what the ICO asks for — you are selling the "
         "control, not the risk.", blocks("NO")],
        ["<b>Professional-body caution.</b> ICAEW is publicly flagging over-reliance on AI "
         "as a reputational threat to the profession.", sev("MEDIUM"),
         "A partner may decline on principle, or wait for guidance.",
         "Lead with governance and approvals. Keep the first workflows away from anything "
         "resembling advice.", blocks("NO")],
        ["<b>Support burden.</b> One person, N single-tenant deployments, no on-call.",
         sev("HIGH"),
         "Caps how many customers you can hold. Roughly 5–8 before it breaks.",
         "Price for it. Business-hours SLA only. Invest in tooling at month 12, not more "
         "customers.", blocks("NO")],
        ["<b>Sales cycle.</b> Professional-services firms are seasonal — January is a "
         "write-off, and so is the run-up to any filing deadline.", sev("MEDIUM"),
         "Three to four months from first contact to signature, with dead months in it.",
         "Build the pipeline before you need it. Avoid launching outreach into January.",
         blocks("NO")],
        ["<b>Upstream dependency on Hermes.</b> ~600k lines you do not control.",
         sev("HIGH"),
         "A breaking upstream change, an abandoned project, or a licence change is an "
         "existential event. A capability already vanished once: the profile allowlist key "
         "NOVA compiled into was removed upstream and had to be probed for at runtime.",
         "Pin the version. Keep the patch budget near zero. Maintain the compatibility "
         "manifest. Watch upstream releases. Contribute so you have standing.", blocks("NO")],
        ["<b>Key-person concentration.</b> One person holds product, sales and delivery.",
         sev("HIGH"),
         "Illness or burnout stops the company. It also caps growth before money does.",
         "Document everything as you go. The runbook and the capability schedule are the "
         "first two artefacts that survive you.", blocks("NO")],
        ["<b>No customers, no case study, no references.</b>", sev("HIGH"),
         "The hardest sale you will ever make is the first one.",
         "Founding-customer pricing with the case study written into the contract. Be "
         "honest that they are first — some buyers value it.", blocks("Partly")],
    ])]

    s += [Gap(3), callout(
        "The four that actually block the first customer",
        "AWS field validation · real provider execution · scheduled execution · the Control "
        "Center exposure decision. <b>Everything else on these two pages can be managed, "
        "disclosed or priced around.</b> These four cannot, and they are the first four "
        "actions in §24. They are also, between them, probably two to three weeks of work — "
        "which is the most encouraging sentence in this document.", "risk")]
    return s


# ---------------------------------------------------------------- PART 24


def part_24():
    s = section("24", "What I should do next", "Fifteen actions, in order")

    s += [P(
        "Ordered by what unblocks the next thing, not by what is most interesting. The "
        "first four are the blockers from §23 and nothing else should be started until they "
        "are done.", "lede")]

    actions = [
        (1, "Deploy to a real AWS account — your own", CRITICAL, "This week",
         "Push the image to ECR, run terraform apply, and prove the six unexercised things: "
         "ECR pull, instance profile, KMS volume mount, awslogs delivery, SSM access, "
         "IMDSv2. Use your own account so the first failures are private. <b>Nothing else "
         "on this list matters until this is done.</b>"),
        (2, "Execute one real model call through Bedrock", CRITICAL, "Same week",
         "Configure a Bedrock model, put the credential in a profile's .env, submit an "
         "objective, and watch a worker complete it. Check token accounting on the Usage "
         "screen against the AWS bill. This is the step that turns a control plane into a "
         "product."),
        (3, "Prove a scheduled job actually fires", CRITICAL, "Week 2",
         "Run a gateway process, create an automation, and watch it execute. Record the "
         "execution ledger row. Until you have seen this, do not sell the Records Chaser."),
        (4, "Settle and document Control Center access", CRITICAL, "Week 2",
         "Choose the loopback-publish plus SSM port-forward route, configure it in the "
         "systemd unit, write the principals file, and confirm the refusals fire when you "
         "get it wrong. Write it into the runbook."),
        (5, "Field-validate email as a channel", SERIOUS, "Week 3",
         "The first offer depends on email. Connect it, send and receive a real message, "
         "and promote it from source_read to field_validated with the evidence. Then one "
         "chat platform."),
        (6, "Drill backup and restore on a fresh host", SERIOUS, "Week 3",
         "Take a backup, destroy the instance, rebuild, restore, verify agents, "
         "automations, knowledge and audit history. Time it, and write the number into the "
         "SLA."),
        (7, "Write the deployment runbook from a real deployment", SERIOUS, "Week 4",
         "Not from §17 of this document — from what actually happened, including what went "
         "wrong. This is the artefact that makes customer two economic."),
        (8, "Confirm the vertical and build the 100-prospect list", ACCENT, "Week 4",
         "UK accountancy practices, 20–150 staff. By hand, with a named partner and a "
         "specific pain hypothesis per row. The research is the qualification."),
        (9, "Build the demo and the 90-second video", ACCENT, "Week 5",
         "A seeded demo tenant that looks like a real practice. Rehearse §13 until it runs "
         "to time without notes. Record the video from the real Control Center."),
        (10, "Publish the governance page and start the LinkedIn cadence", ACCENT, "Week 5",
         "One page, before the rest of the website. Three posts a week from your personal "
         "account. The audience takes eight weeks to build, so start it before you need it."),
        (11, "Contact all 100 and run the discovery calls", ACCENT, "Weeks 6–8",
         "Follow §9. Expect roughly 20 replies, 10 calls, 6 demos. Record every lost reason."),
        (12, "Sign and run the first paid pilot", GOOD, "Weeks 9–10",
         "Founding pricing. Two-week baseline first — refuse to start without it. Case "
         "study written into the agreement."),
        (13, "Measure, review at day 60, and convert", GOOD, "Weeks 11–18",
         "Their numbers. Three honest outcomes on the table. Ask for two referrals in that "
         "meeting, while the number is in front of them."),
        (14, "Write the case study and deploy customer two from the runbook", GOOD,
         "Month 5–6",
         "Named firm, their figures, their quote. Then prove the runbook works by using it "
         "without improvising. Raise to standard pricing."),
        (15, "Onboard the sales partner with the capability schedule", AQUA, "Month 6+",
         "Commission-only, structured per §15, with the one-page annex of what may and may "
         "not be claimed. Not before you have closed three yourself."),
    ]
    for n, title, colour, when, detail in actions:
        body = [
            Paragraph(
                f'<font color="#{colour.hexval()[2:]}" size="14"><b>{n:02d}</b></font>'
                f'&nbsp;&nbsp;<b>{title}</b>&nbsp;&nbsp;'
                f'<font color="#{TEXT_MUTED.hexval()[2:]}" size="7.6">{when}</font>',
                S["h3"]),
            Paragraph(detail, S["cell"]),
        ]
        s.append(Panel(body, fill=PANEL_HI, accent=colour, pad=4.2 * mm))
        s.append(Gap(2.4))

    s += [Gap(2), callout(
        "If you only do three things",
        "<b>Actions 1, 2 and 3.</b> They are probably two weeks of work between them, and "
        "they move NOVA from “a well-built control plane that has never run anywhere real” "
        "to “a product with a deployment.” Every commercial sentence in this document gets "
        "easier to say honestly the moment they are done — and none of them can be said "
        "properly until they are.", "good")]
    return s


# ---------------------------------------------------------------- PART 25


def part_25():
    s = section("25", "Sources and references", "Where every external number came from")

    s += [P(
        "Repository claims are cited inline by file and symbol throughout §1–§3 and are "
        "verifiable at commit <font face='Courier' size='8'>b9319677</font>. External "
        "figures are below. Where a source is self-published by a party with a commercial "
        "interest in the number, that is marked.", "lede")]

    s += [h2("25.1  UK market and AI adoption")]
    s += [table([
        ["Claim used", "Source"],
        ["UK AI adoption ~35% (10+ employees), 29% all bands, ~1.6 technologies per adopter",
         "Office for National Statistics, <i>Artificial intelligence in UK businesses: 2023 "
         "to 2026</i>, published 20 July 2026. <font face='Courier' size='7'>ons.gov.uk/"
         "businessindustryandtrade/business/businessservices/articles/"
         "artificialintelligenceinukbusinesses/2023to2026</font>"],
        ["54% of UK SMEs use AI in some form; 11% use it extensively to automate operations; "
         "95% report no workforce reduction",
         "British Chambers of Commerce, <i>Half of SMEs Using AI</i>, March 2026"],
        ["12% of AI-using businesses report increased revenue; 75% report productivity gains",
         "ONS / DSIT, 2026"],
        ["~5.68m UK SMEs (0–249 employees); 5.64m with 0–49 employees",
         "UK Business Population Estimates 2025 (DBT), via House of Commons Library "
         "research briefing SN06152"],
        ["~29,582 UK accounting &amp; auditing businesses",
         "IBISWorld, <i>Accounting &amp; Auditing in the UK</i>, 2025 <i>(commercial "
         "research)</i>"],
        ["~35,921 active UK accounting companies with recent filings",
         "Firmbase industry list, 2026, derived from Companies House <i>(commercial "
         "aggregator — different definition from IBISWorld)</i>"],
    ], [50 * mm, W - 50 * mm], font_size=7.4)]

    s += [h2("25.2  UK accountancy sector")]
    s += [table([
        ["Claim used", "Source"],
        ["73% of UK accounting firms turning away clients for lack of staff; 74% believe "
         "workloads could push people out of the profession",
         "Advancetrack, <i>Accounting Talent Index</i> 2026"],
        ["95% of mid-tier firms expect increased AI use; 91% expect increased automation; "
         "86% include AI in strategy; 17% believe they can assess AI's workforce impact; "
         "66% investing in upskilling",
         "ICAEW mid-tier firms research, 35 member firms with 11–249 principals, surveyed "
         "February–March 2026"],
        ["ICAEW flagging misuse and over-reliance on AI as a reputational threat to the "
         "profession", "ICAEW insights, 2026"],
        ["Making Tax Digital for Income Tax live from 6 April 2026 for sole traders and "
         "landlords above £50,000 qualifying income",
         "HMRC Making Tax Digital timetable, 2026"],
        ["UK accounts assistant salary ~£23,700–£27,100 depending on source and geography",
         "Multiple 2026 salary trackers; one cites ONS ASHE-derived data at ~£24,993. "
         "<i>Use the customer's own payroll figure in any ROI model.</i>"],
    ], [50 * mm, W - 50 * mm], font_size=7.4)]

    s += [h2("25.3  Pricing and commission benchmarks")]
    s += [table([
        ["Claim used", "Source"],
        ["UK single-workflow automation £1,000–£5,000 build, £200–£800/month support; "
         "connected multi-workflow £4,000–£12,000",
         "UK AI automation agency pricing guides, 2026 <i>(self-published by agencies — "
         "directional only)</i>"],
        ["SMB 2–3 workflows $1,000–$3,500/month; mid-market $4,000–$10,000/month; custom "
         "agent development from $10,000",
         "International AI automation agency pricing surveys, 2026 <i>(self-published)</i>"],
        ["UK marketing agencies spending £350–£1,100 per seat/month on AI tooling",
         "UK agency AI stack reporting, 2026 <i>(self-published)</i>"],
        ["B2B sales commission 5–15% of deal value; SaaS new ARR 8–12%; renewals 2–8%; "
         "pure-commission independent representatives 15–30%",
         "2026 B2B and SaaS commission benchmark surveys (Everstage, CaptivateIQ, Fullcast) "
         "<i>(vendor-published)</i>"],
    ], [50 * mm, W - 50 * mm], font_size=7.4)]

    s += [h2("25.4  Regulatory")]
    s += [table([
        ["Claim used", "Source"],
        ["ICO early views on agentic AI: risk of purposes set too broadly and unfettered "
         "access to data and systems; DPIA warranted where deployment is high-risk; "
         "accountability challenges extend to DPOs",
         "Information Commissioner's Office report on agentic AI, January 2026. The ICO "
         "states it is early-stage thinking and <b>not</b> formal guidance — represent it "
         "that way to customers."],
        ["UK GDPR obligations (lawful basis, data minimisation, purpose limitation, "
         "accuracy, accountability) apply fully to AI systems",
         "ICO guidance on AI and data protection"],
    ], [50 * mm, W - 50 * mm], font_size=7.4)]

    s += [h2("25.5  AWS")]
    s += [callout(
        "No AWS prices are quoted anywhere in this document",
        "The official AWS pricing pages were unreachable from the build environment — the "
        "network egress proxy blocks <font face='Courier' size='7.4'>aws.amazon.com</font>. "
        "Rather than quote figures from memory or from a third-party blog, §18 gives cost "
        "structure and the official URLs. <b>Price every quote against those pages on the "
        "day, and build the estimate in the AWS Pricing Calculator so the customer gets "
        "Amazon's number rather than yours.</b>", "warn")]
    s += [table([
        ["Topic", "Official source"],
        ["EC2 on-demand pricing", "<font face='Courier' size='7'>aws.amazon.com/ec2/pricing/"
         "on-demand/</font>"],
        ["EBS pricing", "<font face='Courier' size='7'>aws.amazon.com/ebs/pricing/</font>"],
        ["ECR pricing", "<font face='Courier' size='7'>aws.amazon.com/ecr/pricing/</font>"],
        ["CloudWatch pricing", "<font face='Courier' size='7'>aws.amazon.com/cloudwatch/"
         "pricing/</font>"],
        ["S3 pricing", "<font face='Courier' size='7'>aws.amazon.com/s3/pricing/</font>"],
        ["Bedrock pricing", "<font face='Courier' size='7'>aws.amazon.com/bedrock/pricing/"
         "</font>"],
        ["KMS pricing", "<font face='Courier' size='7'>aws.amazon.com/kms/pricing/</font>"],
        ["Secrets Manager pricing", "<font face='Courier' size='7'>aws.amazon.com/"
         "secrets-manager/pricing/</font>"],
        ["Pricing calculator", "<font face='Courier' size='7'>calculator.aws</font>"],
        ["Session Manager (SSM) documentation", "<font face='Courier' size='7'>"
         "docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html</font>"],
        ["IMDSv2 documentation", "<font face='Courier' size='7'>docs.aws.amazon.com/"
         "AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html</font>"],
        ["IAM permissions boundaries", "<font face='Courier' size='7'>docs.aws.amazon.com/"
         "IAM/latest/UserGuide/access_policies_boundaries.html</font>"],
    ], [50 * mm, W - 50 * mm], font_size=7.4)]

    s += [h2("25.6  Repository — the primary source for every capability claim")]
    s += [table([
        ["Document", "What it establishes"],
        ["<font face='Courier' size='7'>docs/audits/AWS_DEPLOYMENT_READINESS.md</font>",
         "Image, tag, ports, environment, volumes, AWS services, health check, the 49 local "
         "checks, and the ten remaining deployment requirements."],
        ["<font face='Courier' size='7'>docs/audits/NOVA_HERMES_CAPABILITY_AUDIT.md</font>",
         "The ladder, and what the runtime provides versus what NOVA surfaces. Every row "
         "cites a file and symbol."],
        ["<font face='Courier' size='7'>docs/audits/NOVA_CONTROL_CENTER_GAP_ANALYSIS.md"
         "</font>", "Gaps in order, with what would have to be true to call each validated."],
        ["<font face='Courier' size='7'>docs/audits/PHASE_11_AUTOMATIONS.md</font>, "
         "<font face='Courier' size='7'>PHASE_12_GOVERNED_AUTOMATIONS.md</font>",
         "The scheduler, why create was withheld and then governed, and the ticker "
         "limitation."],
        ["<font face='Courier' size='7'>docs/platform/PRODUCTION_READINESS_AUDIT.md</font>",
         "Twenty findings and their current status."],
        ["<font face='Courier' size='7'>docs/platform/LIVE_RUN.md</font>",
         "The live worker run and the four seam defects only running it could find."],
        ["<font face='Courier' size='7'>docs/PHASE_9.md</font>, "
         "<font face='Courier' size='7'>docs/NOVA_CHANNEL_SECURITY.md</font>",
         "The channel facade, the verification ladder, and the channel threat model."],
        ["<font face='Courier' size='7'>CONTROL-CENTRE-EXTENSIONS-AUDIT.md</font>",
         "S3 knowledge mirroring, MCP and plugins — measured counts and the OAuth reality."],
        ["<font face='Courier' size='7'>nova/capabilities/catalog.yaml</font>",
         "Machine-readable capability catalogue. The source of truth when this document and "
         "the repository disagree."],
        ["<font face='Courier' size='7'>deploy/aws/</font>, "
         "<font face='Courier' size='7'>deploy/docker/</font>",
         "The Terraform module and the image build, including validate-local.sh."],
    ], [66 * mm, W - 66 * mm], font_size=7.4)]

    s += [Gap(4), Rule(), Gap(3)]
    s += [P(
        "<b>When this document and the repository disagree, the repository is right.</b> "
        "This playbook was generated against commit "
        "<font face='Courier' size='8'>b9319677</font> on 15 September 2026. Regenerate it "
        "after any change that moves a capability up or down the ladder in §3 — and update "
        "the sales material and the partner capability schedule at the same time.",
        "body_muted")]
    return s
