const sampleBrand = {
  brandName: "The Pickle Romance",
  category: "Homemade Indian pickles",
  product: "Aam ka Achar",
  price: "Rs. 399",
  offer: "700g Aam ka Achar non-oily jar",
  audience:
    "Working mothers in Faridabad, age 30-40, who like pickles but hesitate because market pickles feel too oily.",
  currentRevenue: "Rs. 10,000",
  revenueGoal: "Rs. 50,000 in 30 days",
  orderChannel: "WhatsApp with UPI before delivery",
  deliveryArea: "Faridabad",
  contentCapacity: "4 Reels per week",
  brandTone: "Homemade, nostalgic, playful, trustworthy",
  channels: ["Instagram Reels", "WhatsApp campaign"],
};

const generatorForm = document.querySelector("#generatorForm");
const loadSample = document.querySelector("#loadSample");
const heroBrandName = document.querySelector("#heroBrandName");
const heroProductName = document.querySelector("#heroProductName");
const heroRevenueTarget = document.querySelector("#heroRevenueTarget");
const heroOrderTarget = document.querySelector("#heroOrderTarget");
const heroPrice = document.querySelector("#heroPrice");
const heroChannel = document.querySelector("#heroChannel");
const heroStatus = document.querySelector("#heroStatus");
const heroCard = document.querySelector("#heroCard");
const briefTitle = document.querySelector("#briefTitle");
const briefCopy = document.querySelector("#briefCopy");
const briefStack = document.querySelector("#briefStack");
const campaignProductName = document.querySelector("#campaignProductName");
const campaignSummary = document.querySelector("#campaignSummary");
const commandCenter = document.querySelector("#commandCenter");
const assetTabs = document.querySelector("#assetTabs");
const assetCard = document.querySelector("#assetCard");
const trackerForm = document.querySelector("#trackerForm");
const insightCard = document.querySelector("#insightCard");

let currentPlan = null;
let activeAssetType = "reels";

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getFormInput() {
  const data = new FormData(generatorForm);
  return {
    brandName: data.get("brandName")?.toString().trim() || "",
    category: data.get("category")?.toString().trim() || "",
    product: data.get("product")?.toString().trim() || "",
    price: data.get("price")?.toString().trim() || "",
    offer: data.get("offer")?.toString().trim() || "",
    audience: data.get("audience")?.toString().trim() || "",
    currentRevenue: data.get("currentRevenue")?.toString().trim() || "",
    revenueGoal: data.get("revenueGoal")?.toString().trim() || "",
    orderChannel: data.get("orderChannel")?.toString().trim() || "",
    deliveryArea: data.get("deliveryArea")?.toString().trim() || "",
    contentCapacity: data.get("contentCapacity")?.toString().trim() || "",
    brandTone: data.get("brandTone")?.toString().trim() || "",
    channels: data.getAll("channels").map((channel) => channel.toString()),
  };
}

function numberFromCurrency(value) {
  const match = value.match(/(?:rs\.?|₹)\s*([0-9,]+)/i);
  return match ? Number(match[1].replaceAll(",", "")) : 0;
}

function compactGoal(value) {
  const number = numberFromCurrency(value);
  if (!number) return "--";
  if (number >= 100000) return `₹${Math.round(number / 100000)}L`;
  if (number >= 1000) return `₹${Math.round(number / 1000)}K`;
  return `₹${number}`;
}

function unitTarget(goal, price) {
  const goalNumber = numberFromCurrency(goal);
  const priceNumber = numberFromCurrency(price);
  if (!goalNumber || !priceNumber) return "add goal + price";
  return `${Math.ceil(goalNumber / priceNumber)} units`;
}

function updateDraftState() {
  const input = getFormInput();
  const hasProduct = Boolean(input.product || input.brandName || input.revenueGoal);
  const primaryChannel = input.channels[0] || "--";

  heroBrandName.textContent = input.brandName || "No brand selected";
  heroProductName.textContent = input.product || "Create a plan";
  heroRevenueTarget.textContent = compactGoal(input.revenueGoal);
  heroOrderTarget.textContent = unitTarget(input.revenueGoal, input.price);
  heroPrice.textContent = input.price || "--";
  heroChannel.textContent = primaryChannel.replace(" campaign", "");
  heroStatus.textContent = currentPlan ? "Live" : "Draft";
  heroCard.classList.toggle("empty-card", !hasProduct && !currentPlan);

  briefTitle.textContent = input.product
    ? `${input.product} growth brief`
    : "Waiting for inputs";
  briefCopy.textContent = input.product
    ? `${input.brandName || "This brand"} wants to grow ${input.product} using ${input.channels.join(", ") || "selected channels"}.`
    : "Add a product and revenue goal. The command center will unlock after AI generation.";

  briefStack.innerHTML = [
    ["Goal", input.revenueGoal],
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

  if (!currentPlan) {
    campaignProductName.textContent = "No active campaign";
    campaignSummary.textContent =
      "Generate a plan to see diagnosis, weekly sprints, experiments, and next actions.";
  }
}

function fillSample() {
  for (const [key, value] of Object.entries(sampleBrand)) {
    if (key === "channels") continue;
    const field = generatorForm.elements.namedItem(key);
    if (field) field.value = value;
  }

  generatorForm.querySelectorAll('input[name="channels"]').forEach((checkbox) => {
    checkbox.checked = sampleBrand.channels.includes(checkbox.value);
  });

  currentPlan = null;
  renderEmptyPlan();
  updateDraftState();
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
  commandCenter.className = "command-empty";
  commandCenter.innerHTML = `
    <h3>No plan generated yet</h3>
    <p>This area will become your weekly growth operating room after you click Generate growth system.</p>
  `;
  assetTabs.innerHTML = "";
  assetCard.className = "asset-card empty-state";
  assetCard.innerHTML = `
    <h3>No assets yet</h3>
    <p>Generate a growth system to unlock copy-ready content.</p>
  `;
}

function renderPlan(plan, input) {
  currentPlan = normalizePlan(plan);
  const summary = currentPlan.summary || {};

  heroStatus.textContent = "Live";
  campaignProductName.textContent = `${input.product} campaign`;
  campaignSummary.textContent =
    summary.positioning ||
    `A 30-day plan to grow ${input.product} through ${input.channels.join(", ")}.`;

  renderCommandCenter(currentPlan, input);
  renderAssetTabs(currentPlan.contentAssets || {});
  updateInsights();
}

function renderCommandCenter(plan, input) {
  const summary = plan.summary || {};
  const diagnosis = plan.diagnosis || [];
  const weeks = plan.weeklyPlan || [];
  const nextActions = plan.nextActions || [];

  commandCenter.className = "command-center";
  commandCenter.innerHTML = `
    <section class="plan-brief">
      <article>
        <span>Positioning</span>
        <strong>${escapeHtml(summary.positioning || "Positioning pending")}</strong>
      </article>
      <article>
        <span>Target</span>
        <strong>${escapeHtml(summary.primaryGoal || input.revenueGoal)}</strong>
      </article>
      <article>
        <span>Unit math</span>
        <strong>${escapeHtml(summary.unitTarget || unitTarget(input.revenueGoal, input.price))}</strong>
      </article>
      <article>
        <span>Risk</span>
        <strong>${escapeHtml(summary.risk || "Track conversion, not just reach.")}</strong>
      </article>
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

    <section class="weekly-board">
      ${weeks.map(renderWeek).join("")}
    </section>

    <section class="next-actions">
      <div>
        <span>Next 7 days</span>
        <h3>Do these first</h3>
      </div>
      <ol>
        ${nextActions.map((action) => `<li>${escapeHtml(action)}</li>`).join("")}
      </ol>
    </section>
  `;
}

function renderWeek(week) {
  const experiments = week.experiments || [];
  return `
    <article class="week-card product-week">
      <div class="week-meta">
        <span>${escapeHtml(week.week || "Week")}</span>
        <strong>${escapeHtml(week.theme || "Growth sprint")}</strong>
        <p>${escapeHtml(week.target || "")}</p>
      </div>
      <div>
        <p>${escapeHtml(week.objective || "")}</p>
        <div class="experiment-grid">
          ${experiments
            .map(
              (experiment) => `
                <div class="experiment">
                  <span>${escapeHtml(experiment.type || "Experiment")}</span>
                  <strong>${escapeHtml(experiment.title || "")}</strong>
                  <p>${escapeHtml(experiment.action || experiment.why || "")}</p>
                  <small>${escapeHtml(experiment.metric || "")}</small>
                </div>
              `,
            )
            .join("")}
        </div>
      </div>
    </article>
  `;
}

function renderAssetTabs(contentAssets) {
  const labels = {
    reels: "Reels",
    whatsapp: "WhatsApp",
    website: "Website",
    linkedin: "LinkedIn",
  };
  const available = Object.keys(labels).filter(
    (key) => Array.isArray(contentAssets[key]) && contentAssets[key].length,
  );

  if (!available.length) {
    assetTabs.innerHTML = "";
    assetCard.className = "asset-card empty-state";
    assetCard.innerHTML = `
      <h3>No assets returned</h3>
      <p>Try regenerating with at least one content channel selected.</p>
    `;
    return;
  }

  if (!available.includes(activeAssetType)) activeAssetType = available[0];

  assetTabs.innerHTML = available
    .map(
      (key) => `
        <button class="tab ${key === activeAssetType ? "active" : ""}" data-asset="${key}">
          ${labels[key]}
        </button>
      `,
    )
    .join("");

  renderAssetCard(contentAssets);
}

function renderAssetCard(contentAssets) {
  const assets = contentAssets[activeAssetType] || [];
  const first = assets[0] || {};
  const body = first.script || first.message || first.copy || first.post || "";

  assetCard.className = "asset-card";
  assetCard.innerHTML = `
    <h3>${escapeHtml(first.title || "Generated asset")}</h3>
    <div class="copy-block">${escapeHtml(body)}</div>
    <button class="copy-btn" data-copy-text="${escapeHtml(body)}">Copy asset</button>
    ${
      assets.length > 1
        ? `<div class="asset-list">${assets
            .slice(1)
            .map(
              (asset) => `
                <article>
                  <strong>${escapeHtml(asset.title || "Asset")}</strong>
                  <p>${escapeHtml(asset.hook || asset.message || asset.copy || asset.post || "")}</p>
                </article>
              `,
            )
            .join("")}</div>`
        : ""
    }
  `;
}

async function generatePlan(event) {
  event.preventDefault();
  const input = getFormInput();

  updateDraftState();
  commandCenter.className = "command-empty loading-state";
  commandCenter.innerHTML = `
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

    renderPlan(data.plan, input);
    document.querySelector("#command")?.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    commandCenter.className = "command-empty error-state";
    commandCenter.innerHTML = `
      <h3>Generation failed</h3>
      <p>${escapeHtml(error.message)}</p>
      <p>Check Anthropic, Vercel environment variables, and Supabase table setup.</p>
    `;
  }
}

function updateInsights() {
  const data = new FormData(trackerForm);
  const views = Number(data.get("views")) || 0;
  const inquiries = Number(data.get("inquiries")) || 0;
  const orders = Number(data.get("orders")) || 0;
  const revenue = Number(data.get("revenue")) || 0;
  const input = getFormInput();
  const goal = numberFromCurrency(input.revenueGoal);
  const inquiryRate = views ? ((inquiries / views) * 100).toFixed(2) : "0.00";
  const conversionRate = inquiries ? ((orders / inquiries) * 100).toFixed(1) : "0.0";
  const targetProgress = goal ? Math.min(100, Math.round((revenue / goal) * 100)) : 0;

  const nextAction =
    orders === 0
      ? "Start by generating a campaign and tracking first inquiries."
      : Number(conversionRate) >= 25
        ? "Conversion is promising. Repeat the strongest hook with clearer offer urgency."
        : "Improve trust proof and WhatsApp CTA before asking for the order.";

  insightCard.innerHTML = `
    <h3>${targetProgress}% of goal</h3>
    <p>Inquiry rate is ${inquiryRate}% and inquiry-to-order conversion is ${conversionRate}%.</p>
    <ul>
      <li><strong>Revenue tracked:</strong> Rs. ${revenue.toLocaleString("en-IN")}</li>
      <li><strong>Orders tracked:</strong> ${orders}</li>
      <li><strong>Next move:</strong> ${nextAction}</li>
    </ul>
  `;
}

document.addEventListener("click", async (event) => {
  const jump = event.target.closest("[data-jump]");
  if (jump) {
    document.querySelector(`#${jump.dataset.jump}`)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  const tab = event.target.closest("[data-asset]");
  if (tab && currentPlan) {
    activeAssetType = tab.dataset.asset;
    renderAssetTabs(currentPlan.contentAssets || {});
  }

  const copy = event.target.closest("[data-copy-text]");
  if (copy) {
    await navigator.clipboard.writeText(copy.dataset.copyText || "");
    copy.textContent = "Copied";
    setTimeout(() => {
      copy.textContent = "Copy asset";
    }, 1400);
  }
});

loadSample.addEventListener("click", fillSample);
generatorForm.addEventListener("input", () => {
  currentPlan = null;
  renderEmptyPlan();
  updateDraftState();
});
generatorForm.addEventListener("submit", generatePlan);
trackerForm.addEventListener("input", updateInsights);

renderEmptyPlan();
updateDraftState();
updateInsights();
