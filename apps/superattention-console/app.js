const sampleBrand = {
  brandName: "The Pickle Romance",
  category: "Homemade Indian pickles",
  product: "Aam ka Achar",
  price: "399",
  offer: "700g Aam ka Achar non-oily jar",
  audience:
    "Working mothers in Faridabad, age 30-40. They like pickles but hesitate because market pickles feel too oily.",
  currentRevenue: "10000",
  goalAmount: "50000",
  goalDays: "30",
  orderChannel: "WhatsApp with UPI before delivery",
  deliveryArea: "Faridabad",
  contentCapacity: "4 Reels per week",
  brandTone: "Homemade, nostalgic, playful, trustworthy",
  brandMission:
    "Indian meals have always had a love affair with pickles. We bottle that romance with ghar-ka taste and less oil.",
  brandDifferentiation:
    "Non-oily and zero-oil lines. Mishri not refined sugar. Founder delivers in Faridabad.",
  brandStage: "Early traction — under Rs 1 lakh/month",
  pastFailures: "Discount posts got views but almost no WhatsApp orders.",
  channels: ["Instagram Reels", "WhatsApp campaign"],
};

const generatorForm = document.querySelector("#generatorForm");
const loadSample = document.querySelector("#loadSample");
const sidebarNav = document.querySelector("#sidebarNav");
const commandCenter = document.querySelector("#commandCenter");
const contentBento = document.querySelector("#contentBento");
const trackerForm = document.querySelector("#trackerForm");
const insightCard = document.querySelector("#insightCard");
const campaignHistoryList = document.querySelector("#campaignHistoryList");
const experimentPanel = document.querySelector("#experimentPanel");
const experimentPanelBody = document.querySelector("#experimentPanelBody");

let currentPlan = null;
let currentCampaignId = null;
let campaignHistory = [];
let activeView = "dashboard";

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatRupee(amount) {
  const n = Number(amount);
  if (!Number.isFinite(n)) return "";
  return `Rs. ${n.toLocaleString("en-IN")}`;
}

function parseNumber(value) {
  if (value == null) return 0;
  const cleaned = String(value).replace(/,/g, "").match(/[0-9]+(\.[0-9]+)?/);
  return cleaned ? Number(cleaned[0]) : 0;
}

function numberFromCurrency(value) {
  return parseNumber(value);
}

function compactGoal(value) {
  const number = numberFromCurrency(value);
  if (!number) return "--";
  if (number >= 100000) return `₹${Math.round(number / 100000)}L`;
  if (number >= 1000) return `₹${Math.round(number / 1000)}K`;
  return `₹${number.toLocaleString("en-IN")}`;
}

function unitTarget(goal, price) {
  const goalNumber = numberFromCurrency(goal);
  const priceNumber = numberFromCurrency(price);
  if (!goalNumber || !priceNumber) return 0;
  return Math.ceil(goalNumber / priceNumber);
}

function buildRevenueGoalString(goalAmount, goalDays) {
  const amount = parseNumber(goalAmount);
  const days = parseNumber(goalDays) || 30;
  if (!amount) return "";
  return `${formatRupee(amount)} in ${days} days`;
}

function getFormInput() {
  const data = new FormData(generatorForm);
  const goalAmount = data.get("goalAmount")?.toString().trim() || "";
  const goalDays = data.get("goalDays")?.toString().trim() || "30";
  const priceRaw = data.get("price")?.toString().trim() || "";
  const revenueRaw = data.get("currentRevenue")?.toString().trim() || "";

  return {
    brandName: data.get("brandName")?.toString().trim() || "",
    category: data.get("category")?.toString().trim() || "",
    product: data.get("product")?.toString().trim() || "",
    price: formatRupee(parseNumber(priceRaw)),
    offer: data.get("offer")?.toString().trim() || "",
    audience: data.get("audience")?.toString().trim() || "",
    currentRevenue: formatRupee(parseNumber(revenueRaw)),
    revenueGoal: buildRevenueGoalString(goalAmount, goalDays),
    goalAmount,
    goalDays,
    orderChannel: data.get("orderChannel")?.toString().trim() || "",
    deliveryArea: data.get("deliveryArea")?.toString().trim() || "",
    contentCapacity: data.get("contentCapacity")?.toString().trim() || "",
    brandTone: data.get("brandTone")?.toString().trim() || "",
    brandMission: data.get("brandMission")?.toString().trim() || "",
    brandDifferentiation: data.get("brandDifferentiation")?.toString().trim() || "",
    brandStage: data.get("brandStage")?.toString().trim() || "",
    pastFailures: data.get("pastFailures")?.toString().trim() || "",
    channels: data.getAll("channels").map((channel) => channel.toString()),
  };
}

function validateFormInput(input) {
  const errors = [];
  if (input.audience.length < 20) {
    errors.push("Audience: describe who, where, and their hesitation (20+ characters).");
  }
  if (input.brandMission.length < 20) {
    errors.push("Brand story: explain why your brand exists (20+ characters).");
  }
  if (input.brandDifferentiation.length < 10) {
    errors.push("Differentiation: say what makes you different (10+ characters).");
  }
  if (!parseNumber(input.goalAmount)) {
    errors.push("Revenue goal: enter a target amount in rupees.");
  }
  if (!parseNumber(input.price)) {
    errors.push("Price: enter a valid selling price.");
  }
  if (!input.channels.length) {
    errors.push("Select at least one marketing channel.");
  }
  return errors;
}

function applyInputToForm(input) {
  const map = {
    brandName: input.brandName,
    category: input.category,
    product: input.product,
    offer: input.offer,
    audience: input.audience,
    orderChannel: input.orderChannel,
    deliveryArea: input.deliveryArea,
    contentCapacity: input.contentCapacity,
    brandTone: input.brandTone,
    brandMission: input.brandMission,
    brandDifferentiation: input.brandDifferentiation,
    brandStage: input.brandStage,
    pastFailures: input.pastFailures || "",
    goalDays: input.goalDays || "30",
  };

  for (const [key, value] of Object.entries(map)) {
    const field = generatorForm.elements.namedItem(key);
    if (field && value != null) field.value = value;
  }

  const priceField = generatorForm.elements.namedItem("price");
  if (priceField) priceField.value = String(parseNumber(input.price) || input.goalAmount || "");

  const revenueField = generatorForm.elements.namedItem("currentRevenue");
  if (revenueField) revenueField.value = String(parseNumber(input.currentRevenue) || "");

  const goalField = generatorForm.elements.namedItem("goalAmount");
  if (goalField) {
    goalField.value = String(parseNumber(input.goalAmount) || parseNumber(input.revenueGoal) || "");
  }

  const channels = input.channels || [];
  generatorForm.querySelectorAll('input[name="channels"]').forEach((checkbox) => {
    checkbox.checked = channels.includes(checkbox.value);
  });
}

function getTrackerData() {
  const data = new FormData(trackerForm);
  return {
    views: Number(data.get("views")) || 0,
    saves: Number(data.get("saves")) || 0,
    shares: Number(data.get("shares")) || 0,
    inquiries: Number(data.get("inquiries")) || 0,
    whatsappClicks: Number(data.get("whatsappClicks")) || 0,
    orders: Number(data.get("orders")) || 0,
    revenue: Number(data.get("revenue")) || 0,
  };
}

function getCloseLearnings() {
  const data = new FormData(trackerForm);
  return {
    bestHook: data.get("bestHook")?.toString().trim() || "",
    topObjection: data.get("topObjection")?.toString().trim() || "",
    repeat: data.get("repeatNext")?.toString().trim() || "",
    stop: data.get("stopDoing")?.toString().trim() || "",
  };
}

function renderSparkBars(containerId, values, variant = "") {
  const container = document.querySelector(containerId);
  if (!container) return;
  container.className = `spark-bars${variant ? ` ${variant}` : ""}`;
  container.innerHTML = values
    .map((height) => `<span style="height:${height}%"></span>`)
    .join("");
}

function setRingProgress(percent) {
  const ring = document.querySelector("#dashboardRing");
  if (!ring) return;
  const circumference = 552.92;
  const offset = circumference - (Math.min(100, percent) / 100) * circumference;
  ring.style.strokeDashoffset = offset;
}

function switchView(view) {
  activeView = view;
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.dataset.view === view);
  });
  document.querySelectorAll(".nav-link[data-view]").forEach((link) => {
    link.classList.toggle("active", link.dataset.view === view);
  });
  if (view === "campaign") {
    void loadCampaignHistory();
  }
}

function statusBadge(status) {
  const normalized = (status || "live").toLowerCase();
  const cls =
    normalized === "closed" ? "status-closed" : normalized === "draft" ? "status-draft" : "status-live";
  return `<span class="history-status ${cls}">${escapeHtml(normalized)}</span>`;
}

function formatHistoryDate(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

async function loadCampaignHistory() {
  if (!campaignHistoryList) return;
  campaignHistoryList.innerHTML = `<p class="muted">Loading past campaigns…</p>`;

  try {
    const response = await fetch("/api/campaigns");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to load history");

    campaignHistory = data.campaigns || [];
    if (!data.configured) {
      campaignHistoryList.innerHTML = `<p class="muted">History unavailable — configure Supabase on Vercel.</p>`;
      return;
    }

    if (!campaignHistory.length) {
      campaignHistoryList.innerHTML = `<p class="muted">No saved campaigns yet. Generate your first plan to start history.</p>`;
      return;
    }

    campaignHistoryList.innerHTML = campaignHistory
      .map(
        (row) => `
          <article class="history-row ${row.id === currentCampaignId ? "active-row" : ""}">
            <div class="history-row-main">
              <strong>${escapeHtml(row.product)}</strong>
              <span class="muted">${escapeHtml(row.brand_name)} · ${formatHistoryDate(row.created_at)}</span>
              <span class="muted">${escapeHtml(row.goal || "")}</span>
            </div>
            <div class="history-row-actions">
              ${statusBadge(row.status)}
              <button class="btn-ghost small" type="button" data-load-campaign="${escapeHtml(row.id)}">Load</button>
              <button class="btn-ghost small" type="button" data-duplicate-campaign="${escapeHtml(row.id)}">Duplicate</button>
            </div>
          </article>
        `,
      )
      .join("");
  } catch (error) {
    campaignHistoryList.innerHTML = `<p class="muted">Could not load history: ${escapeHtml(error.message)}</p>`;
  }
}

function parsePlanFromRow(row) {
  if (!row?.plan_text) return null;
  try {
    return JSON.parse(row.plan_text);
  } catch {
    return null;
  }
}

function loadCampaignFromHistory(id, { duplicate = false } = {}) {
  const row = campaignHistory.find((item) => item.id === id);
  if (!row) return;

  const input = row.input_payload || {};
  applyInputToForm(input);

  if (duplicate) {
    currentPlan = null;
    currentCampaignId = null;
    renderEmptyPlan();
    updateDraftState();
    switchView("setup");
    return;
  }

  const plan = parsePlanFromRow(row);
  if (!plan) {
    alert("This campaign has no readable plan data.");
    return;
  }

  currentCampaignId = row.id;
  renderPlan(plan, getFormInput());

  const metrics = row.metrics?.tracker;
  if (metrics) {
    for (const [key, value] of Object.entries(metrics)) {
      const field = trackerForm.elements.namedItem(key);
      if (field) field.value = value;
    }
  }

  const learnings = row.metrics?.learnings;
  if (learnings) {
    const map = {
      bestHook: learnings.bestHook,
      topObjection: learnings.topObjection,
      repeatNext: learnings.repeat,
      stopDoing: learnings.stop,
    };
    for (const [key, value] of Object.entries(map)) {
      const field = trackerForm.elements.namedItem(key);
      if (field && value) field.value = value;
    }
  }

  switchView("campaign");
  updateDraftState();
}

function updateDraftState() {
  const input = getFormInput();
  const tracker = getTrackerData();
  const goal = numberFromCurrency(input.revenueGoal);
  const units = unitTarget(input.revenueGoal, input.price);
  const progress = goal ? Math.min(100, Math.round((tracker.revenue / goal) * 100)) : 0;
  const primaryChannel = input.channels[0]?.replace(" campaign", "") || "--";
  const isClosed = campaignHistory.find((c) => c.id === currentCampaignId)?.status === "closed";

  document.querySelector("#topBrandName").textContent = input.brandName || "No brand selected";
  document.querySelector("#topGoalChip").textContent = input.revenueGoal
    ? `Current Goal: ${compactGoal(input.revenueGoal)}`
    : "Current Goal: --";
  document.querySelector("#sideGoalValue").textContent = compactGoal(input.revenueGoal);
  document.querySelector("#sideGoalBar").style.width = `${currentPlan ? Math.max(8, progress) : 0}%`;
  document.querySelector("#sideGoalTip").textContent = currentPlan
    ? `AI Insights: ${input.channels.includes("WhatsApp campaign") ? "WhatsApp" : "Primary channel"} shows strongest conversion potential for ${input.product || "this product"}.`
    : "Complete setup to see AI probability insights.";

  document.querySelector("#briefTitle").textContent = input.product
    ? `${input.product} growth brief`
    : "Waiting for inputs";
  document.querySelector("#briefCopy").textContent = input.product
    ? `${input.brandName || "This brand"} wants to grow ${input.product} using ${input.channels.join(", ") || "selected channels"}.`
    : "Add a product and revenue goal. The command center unlocks after AI generation.";

  document.querySelector("#briefStack").innerHTML = [
    ["Goal", input.revenueGoal],
    ["Stage", input.brandStage],
    ["Differentiation", input.brandDifferentiation],
    ["Audience", input.audience],
    ["Order flow", input.orderChannel],
  ]
    .filter(([, value]) => value)
    .map(
      ([label, value]) => `
        <div>
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");

  document.querySelector("#dashboardGoalLabel").textContent = units
    ? `Goal: ${units} units`
    : "Goal: add setup";
  document.querySelector("#dashboardRevenueLabel").textContent = `Target revenue: ${compactGoal(input.revenueGoal)}`;
  document.querySelector("#dashboardUnits").textContent = String(units || 0);
  setRingProgress(progress);

  document.querySelector("#metricViews").textContent = tracker.views.toLocaleString("en-IN");
  document.querySelector("#metricInquiries").textContent = tracker.inquiries.toLocaleString("en-IN");
  document.querySelector("#metricOrders").textContent = tracker.orders.toLocaleString("en-IN");
  document.querySelector("#metricViewsDelta").textContent = tracker.views ? "+12%" : "--";
  document.querySelector("#metricInquiriesDelta").textContent = tracker.inquiries ? "+8%" : "--";
  document.querySelector("#metricOrdersDelta").textContent = tracker.orders ? "+2%" : "--";

  renderSparkBars("#sparkViews", [30, 45, 60, 85]);
  renderSparkBars("#sparkInquiries", [20, 55, 40, 75], "tertiary");
  renderSparkBars("#sparkOrders", [40, 35, 50, 90], "secondary");

  document.querySelector("#dashboardProductName").textContent = currentPlan
    ? `${input.product} campaign`
    : input.product || "No active campaign";
  document.querySelector("#dashboardPositioning").textContent =
    currentPlan?.summary?.positioning ||
    (input.product
      ? `Configure and generate a 30-day plan for ${input.product}.`
      : "Configure brand setup and generate a plan.");
  document.querySelector("#dashboardLiveBadge").textContent = isClosed
    ? "Closed"
    : currentPlan
      ? "Active Campaign"
      : "Draft";
  document.querySelector("#dashboardStatusPill").textContent = isClosed ? "CLOSED" : currentPlan ? "LIVE" : "SETUP";
  document.querySelector("#dashStatGoal").textContent = compactGoal(input.revenueGoal);
  document.querySelector("#dashStatChannel").textContent = primaryChannel;
  document.querySelector("#dashStatRevenue").textContent = `₹${tracker.revenue.toLocaleString("en-IN")}`;
  document.querySelector("#dashStatProgress").textContent = `${progress}%`;

  document.querySelector("#chartGoalLabel").textContent = goal
    ? `Gap analysis against target: ${compactGoal(input.revenueGoal)}`
    : "Gap analysis against target";

  if (!currentPlan) {
    document.querySelector("#campaignProductName").textContent = "No active campaign";
    document.querySelector("#campaignSummary").textContent =
      "Generate a plan to see diagnosis, weekly sprints, and next actions.";
    document.querySelector("#campaignProjection").textContent = "AI Projection: pending";
    document.querySelector("#contentStudioTitle").textContent = "Content Studio";
    document.querySelector("#contentAiBadge").textContent = "AI Content pending";
    document.querySelector("#contentUpdated").textContent = "Generate a plan first";
    document.querySelector("#insightTitle").textContent = "Start with Brand Setup";
    document.querySelector("#insightBody").textContent =
      "Enter your product, audience, and revenue goal. superattention.ai will build a 30-day growth command center—not a document dump.";
  }

  updateInsights();
}

function fillSample() {
  applyInputToForm({
    ...sampleBrand,
    price: sampleBrand.price,
    currentRevenue: sampleBrand.currentRevenue,
    goalAmount: sampleBrand.goalAmount,
  });

  generatorForm.querySelectorAll('input[name="channels"]').forEach((checkbox) => {
    checkbox.checked = sampleBrand.channels.includes(checkbox.value);
  });

  currentPlan = null;
  currentCampaignId = null;
  renderEmptyPlan();
  updateDraftState();
  switchView("setup");
}

function normalizePlan(payload) {
  if (typeof payload === "string") {
    try {
      return JSON.parse(payload);
    } catch {
      return {
        summary: {
          positioning: "Generated text plan",
          primaryGoal: "",
          unitTarget: "",
          coreInsight: payload.slice(0, 240),
          primaryChannel: "",
          risk: "AI returned unstructured text.",
        },
        diagnosis: [],
        weeklyPlan: [],
        contentAssets: {},
        metrics: [],
        nextActions: [],
        rawText: payload,
      };
    }
  }
  return payload || {};
}

function renderEmptyPlan() {
  commandCenter.className = "glass-card command-empty";
  commandCenter.innerHTML = `
    <span class="material-symbols-outlined empty-icon">campaign</span>
    <h3>No plan generated yet</h3>
    <p>Complete Brand Setup and click Generate growth system to unlock your 30-day roadmap.</p>
    <button class="btn-primary" data-view="setup" type="button">Go to Brand Setup</button>
  `;

  contentBento.innerHTML = `
    <div class="glass-card empty-studio">
      <span class="material-symbols-outlined empty-icon">movie_edit</span>
      <h3>No assets yet</h3>
      <p>Reels, WhatsApp, website, and LinkedIn copy appear here after plan generation.</p>
    </div>
  `;
}

function renderPlan(plan, input) {
  currentPlan = normalizePlan(plan);
  const summary = currentPlan.summary || {};

  document.querySelector("#campaignProductName").textContent = `${input.product} campaign`;
  document.querySelector("#campaignSummary").textContent =
    summary.positioning ||
    `A 30-day plan to grow ${input.product} through ${input.channels.join(", ")}.`;
  document.querySelector("#campaignProjection").textContent = "AI Projection: on track";
  document.querySelector("#contentStudioTitle").textContent = `${input.product} Campaign`;
  document.querySelector("#contentAiBadge").textContent = "AI Content Generation Active";
  document.querySelector("#contentUpdated").textContent = "Just generated";

  const insight = summary.coreInsight || currentPlan.nextActions?.[0];
  if (insight) {
    document.querySelector("#insightTitle").textContent = "Execute your top growth move";
    document.querySelector("#insightBody").textContent = insight;
  }

  renderCommandCenter(currentPlan, input);
  renderContentStudio(currentPlan.contentAssets || {}, input);
  updateDraftState();
  void loadCampaignHistory();
}

function renderExperimentDetail(exp) {
  return `
    <article class="experiment-detail">
      <div class="experiment-detail-head">
        <span class="label">${escapeHtml(exp.type || "Experiment")}</span>
        <strong>${escapeHtml(exp.title || "Untitled")}</strong>
      </div>
      <dl>
        <div><dt>Why</dt><dd>${escapeHtml(exp.why || "—")}</dd></div>
        <div><dt>Action</dt><dd>${escapeHtml(exp.action || "—")}</dd></div>
        <div><dt>CTA</dt><dd>${escapeHtml(exp.cta || "—")}</dd></div>
        <div><dt>Metric</dt><dd>${escapeHtml(exp.metric || "—")}</dd></div>
        <div><dt>Decision rule</dt><dd>${escapeHtml(exp.decisionRule || "—")}</dd></div>
      </dl>
    </article>
  `;
}

function openExperimentPanel(weekIndex) {
  const weeks = currentPlan?.weeklyPlan || [];
  const week = weeks[weekIndex];
  if (!week) return;

  document.querySelector("#experimentPanelWeek").textContent = week.week || `Week ${weekIndex + 1}`;
  document.querySelector("#experimentPanelTheme").textContent = week.theme || "Experiments";
  experimentPanelBody.innerHTML = (week.experiments || [])
    .map((exp) => renderExperimentDetail(exp))
    .join("");

  experimentPanel.classList.remove("hidden");
}

function closeExperimentPanel() {
  experimentPanel.classList.add("hidden");
}

function renderCommandCenter(plan, input) {
  const summary = plan.summary || {};
  const diagnosis = plan.diagnosis || [];
  const weeks = plan.weeklyPlan || [];
  const nextActions = plan.nextActions || [];

  const milestones = [
    { day: "DAY 01", title: "Campaign Launch", tone: "primary" },
    { day: "DAY 07", title: "Engagement Peak", tone: "secondary" },
    { day: "DAY 15", title: "Scaling Phase", tone: "tertiary" },
    { day: "DAY 22", title: "Urgency Trigger", tone: "muted" },
    { day: "DAY 30", title: "Final Audit", tone: "muted" },
  ];

  commandCenter.className = "campaign-console";
  commandCenter.innerHTML = `
    <div class="glass-card timeline-card">
      <div class="timeline-header">
        <h4>Strategic Milestones</h4>
        <div class="phase-chips">
          <span class="phase-chip build">Phase 1: Build</span>
          <span class="phase-chip scale">Phase 2: Scale</span>
        </div>
      </div>
      <div class="timeline-track">
        <div class="timeline-progress"></div>
        <div class="timeline-points">
          ${milestones
            .map(
              (m) => `
                <div class="timeline-point ${m.tone}">
                  <div class="dot"></div>
                  <span class="day">${m.day}</span>
                  <span class="title">${m.title}</span>
                </div>
              `,
            )
            .join("")}
        </div>
      </div>
    </div>

    <h4 class="section-label">Weekly Execution Modules</h4>
    <div class="week-grid">
      ${weeks.map((week, index) => renderWeekModule(week, index)).join("")}
    </div>

    <section class="plan-brief">
      <article><span>Positioning</span><strong>${escapeHtml(summary.positioning || "Positioning pending")}</strong></article>
      <article><span>Target</span><strong>${escapeHtml(summary.primaryGoal || input.revenueGoal)}</strong></article>
      <article><span>Unit math</span><strong>${escapeHtml(summary.unitTarget || `${unitTarget(input.revenueGoal, input.price)} units`)}</strong></article>
      <article><span>Risk</span><strong>${escapeHtml(summary.risk || "Track conversion, not just reach.")}</strong></article>
    </section>

    <section class="diagnosis-grid">
      ${diagnosis
        .slice(0, 3)
        .map(
          (item) => `
            <article>
              <span>${escapeHtml(item.label || "Diagnosis")}</span>
              <p>${escapeHtml(item.detail || "")}</p>
            </article>
          `,
        )
        .join("")}
    </section>

    <section class="next-actions-block glass-card">
      <div>
        <span class="label text-primary">Next 7 days</span>
        <h3>Do these first</h3>
      </div>
      <ol>
        ${nextActions.map((action) => `<li>${escapeHtml(action)}</li>`).join("")}
      </ol>
    </section>
  `;
}

function renderWeekModule(week, index) {
  const experiments = week.experiments || [];
  const colors = ["primary", "tertiary", "secondary", "primary"];
  const color = colors[index % colors.length];
  const tasks = experiments.slice(0, 2).map((exp) => exp.title || exp.action || "Experiment");

  return `
    <article class="glass-card week-module">
      <span class="week-num">${String(index + 1).padStart(2, "0")}</span>
      <span class="week-badge">${escapeHtml(week.week || `Week ${index + 1}`)}</span>
      <h5>${escapeHtml(week.theme || "Growth sprint")}</h5>
      <p class="theme">Theme: <em>${escapeHtml(week.objective || week.target || "")}</em></p>
      <div class="week-target">
        <span>Target focus</span>
        <strong>${escapeHtml(week.target || "Grow demand")}</strong>
      </div>
      <ul class="week-tasks">
        ${tasks
          .map(
            (task) => `
              <li>
                <span class="material-symbols-outlined" style="font-size:18px;color:var(--${color});font-variation-settings:'FILL' 1">check_circle</span>
                ${escapeHtml(task)}
              </li>
            `,
          )
          .join("")}
      </ul>
      <button class="week-action" type="button" data-week-index="${index}">View experiments</button>
    </article>
  `;
}

function renderContentStudio(contentAssets, input) {
  const reels = contentAssets.reels || [];
  const whatsapp = contentAssets.whatsapp || [];
  const website = contentAssets.website || [];
  const linkedin = contentAssets.linkedin || [];

  if (!reels.length && !whatsapp.length && !website.length && !linkedin.length) {
    contentBento.innerHTML = `
      <div class="glass-card empty-studio">
        <span class="material-symbols-outlined empty-icon">movie_edit</span>
        <h3>No assets returned</h3>
        <p>Try regenerating with at least one content channel selected.</p>
      </div>
    `;
    return;
  }

  contentBento.innerHTML = `
    ${reels.length ? renderReelsCard(reels) : ""}
    ${whatsapp.length ? renderWhatsappCard(whatsapp) : ""}
    ${website.length ? renderWebsiteCard(website, input) : ""}
    ${linkedin.length ? renderLinkedinCard(linkedin) : ""}
    <div class="glass-card content-card forecast">
      <h5 class="label muted">Growth Forecast</h5>
      <div class="forecast-bars">
        <div class="forecast-row">
          <div class="forecast-row-head"><span>Expected reach</span><strong class="mono-data text-secondary">High</strong></div>
          <div class="bar"><div style="width:75%;background:var(--secondary)"></div></div>
        </div>
        <div class="forecast-row">
          <div class="forecast-row-head"><span>Content synergy</span><strong class="mono-data text-primary">Strong</strong></div>
          <div class="bar"><div style="width:92%;background:var(--primary)"></div></div>
        </div>
      </div>
    </div>
    <div class="glass-card tip-card forecast">
      <h5><span class="material-symbols-outlined">lightbulb</span> AI Content Tip</h5>
      <p>Founder-led and nostalgia hooks usually outperform direct sales posts for food brands. Lead with emotion, close with a clear WhatsApp CTA.</p>
    </div>
  `;
}

function renderReelsCard(reels) {
  return `
    <div class="glass-card content-card reels">
      <div class="content-card-header">
        <h4><span class="material-symbols-outlined text-primary">video_library</span> Reel Hooks</h4>
        <span class="growth-chip">+42% Avg. Engagement</span>
      </div>
      ${reels
        .slice(0, 2)
        .map(
          (asset) => `
            <div class="hook-item">
              <p>"${escapeHtml(asset.hook || asset.script || asset.title || "")}"</p>
              <span class="label muted">${escapeHtml(asset.title || "Hook")}</span>
              <button class="copy-icon-btn" data-copy-text="${escapeHtml(asset.hook || asset.script || "")}" type="button">
                <span class="material-symbols-outlined">content_copy</span>
              </button>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderWhatsappCard(messages) {
  const first = messages[0] || {};
  const text = first.message || first.copy || "";
  return `
    <div class="glass-card content-card whatsapp">
      <div class="content-card-header">
        <h4><span class="material-symbols-outlined text-tertiary">chat</span> WhatsApp Broadcast</h4>
        <span class="label" style="color:var(--tertiary)">Urgency Optimized</span>
      </div>
      <div class="whatsapp-layout">
        <div class="message-block">
          <span class="label muted">Message Content</span>
          <p style="margin-top:12px;font-style:italic">"${escapeHtml(text)}"</p>
          <button class="copy-btn" data-copy-text="${escapeHtml(text)}" type="button">
            <span class="material-symbols-outlined">content_copy</span> Copy Text
          </button>
        </div>
        <div>
          <div class="stat-mini secondary"><strong>84%</strong><span class="label muted">Predicted Open Rate</span></div>
          <div class="stat-mini tertiary" style="margin-top:12px"><strong>12.5%</strong><span class="label muted">Est. Conversion</span></div>
        </div>
      </div>
    </div>
  `;
}

function renderWebsiteCard(pages, input) {
  const first = pages[0] || {};
  const headline = first.headline || first.title || `Taste tradition with ${input.product}`;
  const copy = first.copy || first.message || "";
  return `
    <div class="glass-card content-card website">
      <div class="content-card-header">
        <h4><span class="material-symbols-outlined text-primary">language</span> Website Hero Section</h4>
      </div>
      <div class="hero-preview">
        <h2>${escapeHtml(headline)}</h2>
        <p>${escapeHtml(copy)}</p>
      </div>
      <button class="copy-btn" data-copy-text="${escapeHtml(`${headline}\n\n${copy}`)}" type="button">
        <span class="material-symbols-outlined">content_copy</span> Copy Section Copy
      </button>
    </div>
  `;
}

function renderLinkedinCard(posts) {
  const first = posts[0] || {};
  const body = first.post || first.copy || first.message || "";
  return `
    <div class="glass-card content-card linkedin">
      <div class="content-card-header">
        <h4><span class="material-symbols-outlined text-primary">share</span> LinkedIn Founder Story</h4>
      </div>
      <div class="linkedin-post">
        <p class="label muted">AI Drafted • Founder post</p>
        <p>${escapeHtml(body).replace(/\n/g, "<br>")}</p>
        <button class="copy-btn" data-copy-text="${escapeHtml(body)}" type="button">
          <span class="material-symbols-outlined">content_copy</span> Copy Post
        </button>
      </div>
    </div>
  `;
}

async function generatePlan(event) {
  event.preventDefault();
  const input = getFormInput();
  const errors = validateFormInput(input);

  if (errors.length) {
    alert(errors.join("\n"));
    return;
  }

  updateDraftState();
  switchView("campaign");
  commandCenter.className = "glass-card command-empty loading-state";
  commandCenter.innerHTML = `
    <span class="material-symbols-outlined empty-icon">sync</span>
    <h3>Building your growth system...</h3>
    <p>Creating weekly sprints, channel assets, metrics, and next actions for ${escapeHtml(input.product)}.</p>
  `;

  try {
    const response = await fetch("/api/generate-plan", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(input),
    });
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || "Failed to generate plan");

    if (data.saveResult?.campaign?.id) {
      currentCampaignId = data.saveResult.campaign.id;
    }

    renderPlan(data.plan, input);

    if (data.usedPriorLearnings) {
      document.querySelector("#campaignProjection").textContent =
        "AI Projection: informed by last closed campaign";
    }

    switchView("campaign");
  } catch (error) {
    commandCenter.className = "glass-card command-empty error-state";
    commandCenter.innerHTML = `
      <span class="material-symbols-outlined empty-icon">error</span>
      <h3>Generation failed</h3>
      <p>${escapeHtml(error.message)}</p>
      <p>Check Anthropic, Vercel environment variables, and Supabase table setup.</p>
    `;
  }
}

async function closeCampaign() {
  const statusEl = document.querySelector("#closeCampaignStatus");
  if (!currentCampaignId) {
    statusEl.textContent = "Generate or load a campaign first before closing.";
    return;
  }

  const tracker = getTrackerData();
  const learnings = getCloseLearnings();

  statusEl.textContent = "Saving learnings…";

  try {
    const response = await fetch("/api/campaigns", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        id: currentCampaignId,
        tracker,
        learnings,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to close campaign");

    statusEl.textContent = "Campaign closed. Your next plan will use these learnings.";
    document.querySelector("#dashboardStatusPill").textContent = "CLOSED";
    await loadCampaignHistory();
    updateDraftState();
  } catch (error) {
    statusEl.textContent = error.message;
  }
}

function renderChartBars() {
  const tracker = getTrackerData();
  const input = getFormInput();
  const goal = numberFromCurrency(input.revenueGoal);
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const currentHeights = [55, 65, 45, 80, 95, 30, 10];
  const goalHeight = 70;

  document.querySelector("#chartBars").innerHTML = days
    .map(
      (day, index) => `
        <div class="chart-day">
          <div class="bars">
            <span class="goal" style="height:${goalHeight}%"></span>
            <span class="current" style="height:${currentHeights[index]}%"></span>
          </div>
          <span class="label muted">${day}</span>
        </div>
      `,
    )
    .join("");

  const gap = Math.max(0, goal - tracker.revenue);
  document.querySelector("#revenueGap").textContent = gap
    ? `₹${gap.toLocaleString("en-IN")} left`
    : "Goal reached";
  document.querySelector("#avgOrder").textContent =
    tracker.orders > 0 ? `₹${Math.round(tracker.revenue / tracker.orders).toLocaleString("en-IN")}` : "--";
  document.querySelector("#revenueGrowthChip").textContent =
    tracker.revenue > 0
      ? `+${Math.min(99, Math.round((tracker.revenue / Math.max(goal, 1)) * 100))}% Revenue`
      : "+0% Revenue";
}

function updateInsights() {
  const tracker = getTrackerData();
  const input = getFormInput();
  const goal = numberFromCurrency(input.revenueGoal);
  const inquiryRate = tracker.views ? ((tracker.inquiries / tracker.views) * 100).toFixed(2) : "0.00";
  const conversionRate = tracker.inquiries
    ? ((tracker.orders / tracker.inquiries) * 100).toFixed(1)
    : "0.0";
  const targetProgress = goal ? Math.min(100, Math.round((tracker.revenue / goal) * 100)) : 0;

  const repeat =
    currentPlan?.nextActions?.[0] ||
    "Lead with the strongest hook from your generated Reels and test one new angle each week.";
  const change =
    currentPlan?.summary?.risk ||
    "Improve trust proof and WhatsApp CTA clarity before pushing harder on reach.";

  insightCard.innerHTML = `
    <div class="ai-strategy-header">
      <h4><span class="material-symbols-outlined">auto_awesome</span> AI Strategy</h4>
      <span class="confidence-badge">High Confidence</span>
    </div>
    <div class="ai-strategy-body">
      <div class="strategy-item">
        <div class="strategy-icon repeat"><span class="material-symbols-outlined">replay</span></div>
        <div>
          <span class="label muted">What to repeat</span>
          <p>${escapeHtml(repeat)}</p>
        </div>
      </div>
      <div class="strategy-item">
        <div class="strategy-icon change"><span class="material-symbols-outlined">swap_horiz</span></div>
        <div>
          <span class="label muted">What to change</span>
          <p>${escapeHtml(change)}</p>
        </div>
      </div>
      <p class="muted" style="margin-top:16px">${targetProgress}% of goal • Inquiry rate ${inquiryRate}% • Conversion ${conversionRate}%</p>
    </div>
  `;

  const learnings = getCloseLearnings();
  if (learnings.topObjection) {
    document.querySelector("#objectionQuote").textContent = `"${learnings.topObjection}"`;
  }

  renderChartBars();
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !experimentPanel.classList.contains("hidden")) {
    closeExperimentPanel();
  }
});

document.addEventListener("click", async (event) => {
  const viewTrigger = event.target.closest("[data-view]");
  if (viewTrigger?.dataset.view) {
    switchView(viewTrigger.dataset.view);
  }

  const weekBtn = event.target.closest("[data-week-index]");
  if (weekBtn) {
    openExperimentPanel(Number(weekBtn.dataset.weekIndex));
    return;
  }

  if (event.target.closest("[data-close-experiments]")) {
    closeExperimentPanel();
    return;
  }

  const loadBtn = event.target.closest("[data-load-campaign]");
  if (loadBtn) {
    loadCampaignFromHistory(loadBtn.dataset.loadCampaign);
    return;
  }

  const dupBtn = event.target.closest("[data-duplicate-campaign]");
  if (dupBtn) {
    loadCampaignFromHistory(dupBtn.dataset.duplicateCampaign, { duplicate: true });
    return;
  }

  const copy = event.target.closest("[data-copy-text]");
  if (copy) {
    await navigator.clipboard.writeText(copy.dataset.copyText || "");
    const original = copy.innerHTML;
    copy.innerHTML = '<span class="material-symbols-outlined">check</span> Copied';
    setTimeout(() => {
      copy.innerHTML = original;
    }, 1400);
  }
});

document.querySelector("#optimizeBtn")?.addEventListener("click", () => {
  if (!currentPlan) {
    switchView("setup");
    return;
  }
  switchView("tracker");
});

document.querySelector("#syncMetrics")?.addEventListener("click", () => {
  updateDraftState();
  document.querySelector("#closeCampaignStatus").textContent = "Metrics synced to dashboard.";
});

document.querySelector("#closeCampaignBtn")?.addEventListener("click", () => {
  void closeCampaign();
});

document.querySelector("#refreshHistory")?.addEventListener("click", () => {
  void loadCampaignHistory();
});

sidebarNav?.addEventListener("click", (event) => {
  const link = event.target.closest(".nav-link[data-view]");
  if (link) switchView(link.dataset.view);
});

loadSample.addEventListener("click", fillSample);
generatorForm.addEventListener("input", () => {
  if (!currentCampaignId) {
    currentPlan = null;
    renderEmptyPlan();
  }
  updateDraftState();
});
generatorForm.addEventListener("submit", generatePlan);
trackerForm.addEventListener("input", () => {
  updateDraftState();
  updateInsights();
});

renderEmptyPlan();
updateDraftState();
setRingProgress(0);
void loadCampaignHistory();
