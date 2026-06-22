const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";

function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

function requireEnv(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function buildPrompt(input) {
  return `
You are superattention.ai, an AI growth system for early Indian D2C brands.

Your job is to turn attention into revenue. Think like a senior growth operator who understands content, offers, website conversion, WhatsApp/email campaigns, founder-led LinkedIn, and weekly learning loops.

Create a production-ready 30-day growth plan using the details below.

Brand:
- Name: ${input.brandName}
- Category: ${input.category}
- Current monthly revenue: ${input.currentRevenue}
- Revenue goal: ${input.revenueGoal}
- Brand tone: ${input.brandTone}

Product / offer:
- Product: ${input.product}
- Price: ${input.price}
- Offer: ${input.offer}

Audience and market:
- Target customer: ${input.audience}
- Delivery area: ${input.deliveryArea}
- Order channel: ${input.orderChannel}
- Content capacity: ${input.contentCapacity}

Channels to use:
${input.channels.map((channel) => `- ${channel}`).join("\n")}

Return ONLY valid JSON. Do not wrap it in markdown. Do not include commentary before or after JSON.

Use this exact structure:
{
  "summary": {
    "positioning": "one sharp campaign positioning sentence",
    "primaryGoal": "revenue goal",
    "unitTarget": "estimated units/orders needed",
    "coreInsight": "most important customer insight",
    "primaryChannel": "highest leverage channel",
    "risk": "biggest risk to watch"
  },
  "diagnosis": [
    {"label": "What is working", "detail": "specific diagnosis"},
    {"label": "What is broken", "detail": "specific diagnosis"},
    {"label": "Growth lever", "detail": "specific diagnosis"}
  ],
  "weeklyPlan": [
    {
      "week": "Week 1",
      "theme": "theme name",
      "target": "weekly target",
      "objective": "plain-English objective",
      "experiments": [
        {
          "type": "Reel / WhatsApp / Website / Offer / LinkedIn",
          "title": "short title",
          "why": "why this experiment matters",
          "action": "what founder should do",
          "cta": "exact CTA",
          "metric": "main metric",
          "decisionRule": "what to do based on result"
        }
      ]
    }
  ],
  "contentAssets": {
    "reels": [{"title": "title", "hook": "hook", "script": "short script", "cta": "cta"}],
    "whatsapp": [{"title": "title", "message": "message"}],
    "website": [{"title": "title", "copy": "copy"}],
    "linkedin": [{"title": "title", "post": "post"}]
  },
  "metrics": [
    {"name": "metric", "why": "why it matters", "target": "target"}
  ],
  "nextActions": [
    "specific action for the next 7 days"
  ]
}

Be specific, practical, revenue-focused, and suitable for a founder with limited time and budget.
`.trim();
}

async function callAnthropic(input) {
  const apiKey = requireEnv("ANTHROPIC_API_KEY");
  const model = process.env.ANTHROPIC_MODEL || "claude-3-5-sonnet-latest";

  const response = await fetch(ANTHROPIC_API_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: 4000,
      messages: [
        {
          role: "user",
          content: buildPrompt(input),
        },
      ],
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    const message = data?.error?.message || "Anthropic request failed";
    throw new Error(message);
  }

  const text = data.content
    ?.filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n\n");

  if (!text) {
    throw new Error("Anthropic returned an empty response");
  }

  const cleaned = text
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch (error) {
    throw new Error(`AI returned invalid JSON: ${error.message}`);
  }
}

async function saveCampaign(input, plan) {
  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const table = process.env.SUPABASE_CAMPAIGNS_TABLE || "campaigns";

  if (!supabaseUrl || !serviceRoleKey) {
    return { saved: false, reason: "Supabase environment variables not configured" };
  }

  const response = await fetch(`${supabaseUrl}/rest/v1/${table}`, {
    method: "POST",
    headers: {
      apikey: serviceRoleKey,
      authorization: `Bearer ${serviceRoleKey}`,
      "content-type": "application/json",
      prefer: "return=representation",
    },
    body: JSON.stringify({
      brand_name: input.brandName,
      product: input.product,
      goal: input.revenueGoal,
      audience: input.audience,
      channels: input.channels,
      delivery_area: input.deliveryArea,
      content_capacity: input.contentCapacity,
      plan_text: JSON.stringify(plan, null, 2),
      input_payload: input,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.message || "Failed to save campaign to Supabase");
  }

  return { saved: true, campaign: data[0] };
}

function validateInput(input) {
  const required = [
    "brandName",
    "category",
    "product",
    "price",
    "offer",
    "audience",
    "revenueGoal",
    "orderChannel",
    "deliveryArea",
    "contentCapacity",
    "brandTone",
    "currentRevenue",
  ];

  for (const field of required) {
    if (!input[field] || typeof input[field] !== "string") {
      return `${field} is required`;
    }
  }

  if (!Array.isArray(input.channels) || input.channels.length === 0) {
    return "At least one growth channel is required";
  }

  return null;
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return json(res, 405, { error: "Method not allowed" });
  }

  try {
    const input = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    const validationError = validateInput(input);

    if (validationError) {
      return json(res, 400, { error: validationError });
    }

    const plan = await callAnthropic(input);
    const saveResult = await saveCampaign(input, plan);

    return json(res, 200, {
      plan,
      saveResult,
    });
  } catch (error) {
    return json(res, 500, {
      error: error.message || "Failed to generate plan",
    });
  }
};
