import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface PaperclipContext {
  apiUrl: string;
  apiKey: string;
  companyId: string;
}

/**
 * Paperclip Bridge Skill
 * Allows Hermes to create goals/issues in Paperclip from Telegram DMs
 */

export async function createGoalInPaperclip(
  context: PaperclipContext,
  goal: {
    title: string;
    description: string;
    priority: "high" | "medium" | "low";
  }
): Promise<{ id: string; title: string }> {
  const response = await fetch(
    `${context.apiUrl}/api/companies/${context.companyId}/goals`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${context.apiKey}`,
      },
      body: JSON.stringify({
        title: goal.title,
        description: goal.description,
        level: "company",
        status: "active",
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      `Failed to create goal: ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

export async function createIssueInPaperclip(
  context: PaperclipContext,
  issue: {
    title: string;
    description: string;
    priority: "high" | "medium" | "low";
    assigneeId?: string;
  }
): Promise<{ id: string; title: string }> {
  const response = await fetch(
    `${context.apiUrl}/api/companies/${context.companyId}/issues`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${context.apiKey}`,
      },
      body: JSON.stringify({
        title: issue.title,
        description: issue.description,
        priority: issue.priority,
        status: "todo",
        assigneeAgentId: issue.assigneeId,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      `Failed to create issue: ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

/**
 * Main entrypoint for Hermes to use this skill
 * Call when Seb asks to create a task/goal in Paperclip via DM
 */
export async function handlePaperclipRequest(
  userMessage: string,
  context: PaperclipContext
): Promise<string> {
  // Ask clarifying questions if needed
  const clarificationResponse = await client.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 500,
    system: `You are helping to create a Paperclip task/goal from a user request.
    
Ask up to 3 clarifying questions if the request is vague:
1. What's the goal/objective? (one sentence)
2. What does success look like?
3. Priority: P0 (blocking), P1 (this week), or P2 (backlog)?

If the request has all this info, respond with READY_TO_CREATE and your summary.`,
    messages: [
      {
        role: "user",
        content: userMessage,
      },
    ],
  });

  const clarificationText =
    clarificationResponse.content[0].type === "text"
      ? clarificationResponse.content[0].text
      : "";

  if (!clarificationText.includes("READY_TO_CREATE")) {
    return clarificationText;
  }

  // Extract details and create in Paperclip
  const extractResponse = await client.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 500,
    system: `Extract the goal/task details from this request as JSON:
{
  "title": "string",
  "description": "string",
  "priority": "high|medium|low"
}

Return ONLY the JSON, no other text.`,
    messages: [
      {
        role: "user",
        content: userMessage,
      },
    ],
  });

  const jsonText =
    extractResponse.content[0].type === "text"
      ? extractResponse.content[0].text
      : "{}";
  const taskDetails = JSON.parse(jsonText);

  // Determine if this is a goal (strategic) or issue (task)
  const isGoal =
    userMessage.toLowerCase().includes("goal") ||
    userMessage.toLowerCase().includes("project");

  try {
    let result;
    if (isGoal) {
      result = await createGoalInPaperclip(context, taskDetails);
      return `✅ Goal created in Paperclip: "${result.title}" (ID: ${result.id}). I'll delegate this to the team on the next heartbeat.`;
    } else {
      result = await createIssueInPaperclip(context, taskDetails);
      return `✅ Task created in Paperclip: "${result.title}" (ID: ${result.id}). Assigning to the right agent...`;
    }
  } catch (error) {
    return `❌ Failed to create in Paperclip: ${error instanceof Error ? error.message : "Unknown error"}`;
  }
}
