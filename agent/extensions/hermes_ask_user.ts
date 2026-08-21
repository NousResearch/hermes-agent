/**
 * hermes_ask_user — first-party pi extension for Hermes pi-rpc delegation.
 *
 * Exposes an `hermes_ask_user` tool to the pi agent. When called, it raises a
 * blocking UI dialog over the RPC channel (extension_ui_request). Hermes'
 * PiRPCClient holds the question open, surfaces it to the user as
 * [pi-question], and the user's steering reply is delivered back as the
 * answer.
 *
 * Uses the documented ExtensionUIContext API: ctx.ui.input() / ctx.ui.select(),
 * both of which resolve to string | undefined (undefined = cancelled/timeout).
 */
export default function (pi: any) {
  pi.registerTool({
    name: "hermes_ask_user",
    label: "Ask User (Hermes)",
    description:
      "Ask the supervising human a question and block until they answer. " +
      "Use when you genuinely need information or a decision before proceeding.",
    executionMode: "sequential",
    parameters: {
      type: "object",
      properties: {
        question: { type: "string", description: "The question, self-contained." },
        options: {
          type: "array",
          items: { type: "string" },
          description: "Optional choices; the human may also answer freely.",
        },
      },
      required: ["question"],
    },
    async execute(toolCallId: string, args: { question: string; options?: string[] }, signal: any, onUpdate: any, ctx: any) {
      const question = (args?.question ?? "").trim() || "(no question)";
      const options = (args?.options ?? []).map((o) => String(o).trim()).filter(Boolean);
      const ui = ctx && ctx.ui ? ctx.ui : (pi as any).ui;
      if (!ui || typeof ui.input !== "function") {
        return {
          content: [{ type: "text", text: "(no answer provided — proceed with a safe default and note it)" }],
        };
      }

      let answer: string | undefined;
      if (options.length > 0 && typeof ui.select === "function") {
        // Prepend an explicit freeform escape so the human is never locked in.
        answer = await ui.select(question, [...options, "Other (type a custom answer)"]);
        if (answer === "Other (type a custom answer)") {
          answer = await ui.input(question, "Type your answer...");
        }
      } else {
        answer = await ui.input(question, "Type your answer...");
      }

      const text = (answer ?? "").trim();
      if (!text) {
        return {
          content: [{ type: "text", text: "(no answer provided — proceed with a safe default and note it)" }],
        };
      }
      return { content: [{ type: "text", text: `User answered: ${text}` }] };
    },
  });
}
