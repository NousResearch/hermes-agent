function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

function tableName() {
  return process.env.SUPABASE_CAMPAIGNS_TABLE || "campaigns";
}

function supabaseHeaders(serviceRoleKey, extra = {}) {
  return {
    apikey: serviceRoleKey,
    authorization: `Bearer ${serviceRoleKey}`,
    ...extra,
  };
}

function campaignStatus(metrics) {
  if (metrics && typeof metrics === "object" && metrics.status) {
    return metrics.status;
  }
  return "live";
}

async function listCampaigns() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const table = tableName();

  if (!supabaseUrl || !serviceRoleKey) {
    return {
      campaigns: [],
      configured: false,
      reason: "Supabase environment variables not configured",
    };
  }

  const response = await fetch(
    `${supabaseUrl}/rest/v1/${table}?select=id,created_at,brand_name,product,goal,audience,channels,plan_text,input_payload,metrics&order=created_at.desc&limit=20`,
    {
      headers: supabaseHeaders(serviceRoleKey),
    },
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.message || "Failed to fetch campaigns from Supabase");
  }

  return {
    campaigns: data.map((row) => ({
      ...row,
      status: campaignStatus(row.metrics),
    })),
    configured: true,
  };
}

async function closeCampaign(body) {
  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const table = tableName();

  if (!supabaseUrl || !serviceRoleKey) {
    throw new Error("Supabase environment variables not configured");
  }

  const { id, tracker, learnings } = body;
  if (!id) {
    throw new Error("Campaign id is required");
  }

  const metrics = {
    status: "closed",
    closedAt: new Date().toISOString(),
    tracker: tracker || {},
    learnings: learnings || {},
  };

  const response = await fetch(`${supabaseUrl}/rest/v1/${table}?id=eq.${id}`, {
    method: "PATCH",
    headers: supabaseHeaders(serviceRoleKey, {
      "content-type": "application/json",
      prefer: "return=representation",
    }),
    body: JSON.stringify({ metrics }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.message || "Failed to close campaign");
  }

  return { campaign: data[0] };
}

module.exports = async function handler(req, res) {
  try {
    if (req.method === "GET") {
      const result = await listCampaigns();
      return json(res, 200, result);
    }

    if (req.method === "PATCH") {
      const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
      const result = await closeCampaign(body);
      return json(res, 200, result);
    }

    return json(res, 405, { error: "Method not allowed" });
  } catch (error) {
    return json(res, 500, {
      error: error.message || "Campaign request failed",
    });
  }
};
