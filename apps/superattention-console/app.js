const weeks = [
  {
    label: "Week 1",
    title: "Awareness",
    target: "21 combos",
    summary: "Make customers understand why Aam Romantics is different from oily market pickles.",
    experiments: [
      {
        type: "Reel",
        title: "Oily market achar hesitation",
        copy: "Market wale achar ka oil dekh ke darr lagta hai?",
      },
      {
        type: "Pairing",
        title: "Dal-chawal romance",
        copy: "Dal-chawal ki love story tab complete hoti hai jab achar saath ho.",
      },
      {
        type: "Trust",
        title: "Founder delivery",
        copy: "Faridabad mein founder khud achar deliver karega.",
      },
      {
        type: "Seasonal",
        title: "July mango season",
        copy: "July ka aam season, aur ghar ka achar romance.",
      },
    ],
  },
  {
    label: "Week 2",
    title: "Trust",
    target: "21 combos",
    summary: "Make the brand feel homemade, safe, local, and founder-led.",
    experiments: [
      {
        type: "Proof",
        title: "Non-oily vs zero-oil",
        copy: "Explain which jars are non-oily and which are zero-oil in plain language.",
      },
      {
        type: "Product",
        title: "What's inside Aam Romantics?",
        copy: "Show each jar and the meal moment it completes.",
      },
      {
        type: "Local",
        title: "Faridabad families first",
        copy: "Position the launch as a local community batch.",
      },
      {
        type: "BTS",
        title: "Packing your order",
        copy: "Show clean packing, jar close-ups, and handwritten thank-you notes.",
      },
    ],
  },
  {
    label: "Week 3",
    title: "Conversion",
    target: "24 combos",
    summary: "Turn interested viewers into WhatsApp orders with price clarity and a simple order path.",
    experiments: [
      {
        type: "Offer",
        title: "3 jars, 3 moods",
        copy: "Spicy mango, sweet mango, chatpata hing: one July combo.",
      },
      {
        type: "Price",
        title: "Rs. 599 explained",
        copy: "Make the combo value and delivery rules impossible to miss.",
      },
      {
        type: "Flow",
        title: "Order in 30 seconds",
        copy: "DM AAM, share location, pay by UPI, receive delivery.",
      },
      {
        type: "Quiz",
        title: "Which meal needs which achar?",
        copy: "Pair paratha, dal-chawal, khichdi, and poha with different jars.",
      },
    ],
  },
  {
    label: "Week 4",
    title: "Urgency + Learning",
    target: "18-24 combos",
    summary: "Close the July batch and capture learnings for the next growth sprint.",
    experiments: [
      {
        type: "Urgency",
        title: "Last July batch",
        copy: "Use honest stock updates and clear WhatsApp CTA.",
      },
      {
        type: "Proof",
        title: "Most-loved jar",
        copy: "Share which flavour got the most interest.",
      },
      {
        type: "Poll",
        title: "What should we make next?",
        copy: "Ask followers to choose Nimbu, Lahsun, Karele, or Teekhi Hari Mirch.",
      },
      {
        type: "Founder",
        title: "July learning",
        copy: "Share what customers loved and what the brand is improving.",
      },
    ],
  },
];

const assets = [
  {
    id: "reel",
    label: "Reel Script",
    title: "Oily market achar Reel",
    body: `Hook: Market wale achar ka oil dekh ke darr lagta hai?

Love achar but hate when it is floating in oil?
Meet Aam Romantics by The Pickle Romance.
One non-oily Aam ka Achar.
One zero-oil Aam ka Chunda.
One zero-oil Hing ka Achar.

Made for everyday Indian meals that deserve a little romance.

CTA: DM AAM to order in Faridabad + Delhi NCR.`,
  },
  {
    id: "whatsapp",
    label: "WhatsApp",
    title: "Order reply template",
    body: `Thank you for ordering from The Pickle Romance.

Aam Romantics Combo includes:
- 200g Aam ka Achar
- 200g Aam ka Chunda
- 200g Hing ka Achar

Price: Rs. 599
Delivery: Free in Faridabad. Delivery charges extra for Delhi/Gurgaon/Noida.
Payment: UPI before delivery.

Please share your full address and preferred delivery time.`,
  },
  {
    id: "website",
    label: "Website Copy",
    title: "Landing page hero",
    body: `Bring romance back to everyday Indian meals.

Homemade-style non-oily and zero-oil pickles from The Pickle Romance.

July Special: Aam Romantics Combo
Aam ka Achar + Aam ka Chunda + Hing ka Achar
3 x 200g jars
Rs. 599

Available in Faridabad and Delhi NCR.

CTA: Order on WhatsApp`,
  },
  {
    id: "linkedin",
    label: "LinkedIn",
    title: "Founder-led post",
    body: `I am building The Pickle Romance because Indian meals have always had a love affair with pickles.

But many families avoid market pickles because they feel too oily for everyday meals.

Our July experiment is simple:
Can we sell 84 Aam Romantics combos in 30 days through Reels, WhatsApp, and founder-led local delivery?

The combo:
- Aam ka Achar
- Aam ka Chunda
- Hing ka Achar

Goal: Rs. 50,000 in July.
I will share what works, what fails, and what customers teach us.`,
  },
];

const timeline = document.querySelector("#timeline");
const assetTabs = document.querySelector("#assetTabs");
const assetCard = document.querySelector("#assetCard");
const trackerForm = document.querySelector("#trackerForm");
const insightCard = document.querySelector("#insightCard");

function renderTimeline() {
  timeline.innerHTML = weeks
    .map(
      (week, index) => `
        <article class="week-card" style="animation-delay: ${index * 120}ms">
          <div class="week-meta">
            <span>${week.label}</span>
            <strong>${week.title}</strong>
            <p>${week.target}</p>
          </div>
          <div>
            <p>${week.summary}</p>
            <div class="experiment-grid">
              ${week.experiments
                .map(
                  (experiment) => `
                    <div class="experiment">
                      <span>${experiment.type}</span>
                      <strong>${experiment.title}</strong>
                      <p>${experiment.copy}</p>
                    </div>
                  `,
                )
                .join("")}
            </div>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderTabs(activeId = assets[0].id) {
  assetTabs.innerHTML = assets
    .map(
      (asset) => `
        <button class="tab ${asset.id === activeId ? "active" : ""}" data-asset="${asset.id}">
          ${asset.label}
        </button>
      `,
    )
    .join("");

  const activeAsset = assets.find((asset) => asset.id === activeId) ?? assets[0];
  assetCard.innerHTML = `
    <h3>${activeAsset.title}</h3>
    <div class="copy-block">${activeAsset.body}</div>
    <button class="copy-btn" data-copy="${activeAsset.id}">Copy asset</button>
  `;
}

function updateInsights() {
  const data = new FormData(trackerForm);
  const views = Number(data.get("views")) || 0;
  const inquiries = Number(data.get("inquiries")) || 0;
  const orders = Number(data.get("orders")) || 0;
  const revenue = Number(data.get("revenue")) || 0;
  const inquiryRate = views ? ((inquiries / views) * 100).toFixed(2) : "0.00";
  const conversionRate = inquiries ? ((orders / inquiries) * 100).toFixed(1) : "0.0";
  const targetProgress = Math.min(100, Math.round((revenue / 50000) * 100));

  const nextAction =
    Number(conversionRate) >= 30
      ? "Repeat the same hook with price clarity and founder delivery proof."
      : "Improve the WhatsApp CTA and add trust proof before asking for the order.";

  insightCard.innerHTML = `
    <h3>${targetProgress}% of monthly target</h3>
    <p>Inquiry rate is ${inquiryRate}% and WhatsApp-to-order conversion is ${conversionRate}%.</p>
    <ul>
      <li><strong>Revenue tracked:</strong> Rs. ${revenue.toLocaleString("en-IN")}</li>
      <li><strong>Orders tracked:</strong> ${orders} Aam Romantics combos</li>
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
  if (tab) {
    renderTabs(tab.dataset.asset);
  }

  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) {
    const asset = assets.find((item) => item.id === copyButton.dataset.copy);
    if (!asset) return;
    await navigator.clipboard.writeText(asset.body);
    copyButton.textContent = "Copied";
    setTimeout(() => {
      copyButton.textContent = "Copy asset";
    }, 1400);
  }
});

trackerForm.addEventListener("input", updateInsights);

renderTimeline();
renderTabs();
updateInsights();
