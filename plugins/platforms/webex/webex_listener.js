#!/usr/bin/env node
"use strict";

function emit(type, payload) {
  process.stdout.write(`${JSON.stringify({ type, ...payload })}\n`);
}

function formatError(error) {
  if (!error) {
    return "Unknown Webex listener error";
  }
  if (error instanceof Error) {
    return error.stack || error.message || String(error);
  }
  return String(error);
}

async function main() {
  const token = String(process.env.WEBEX_BOT_TOKEN || "").trim();
  if (!token) {
    emit("fatal", { message: "WEBEX_BOT_TOKEN is required" });
    process.exit(1);
    return;
  }

  process.env.WEBEX_LOG_LEVEL = process.env.WEBEX_LOG_LEVEL || "silent";

  let WebexCore;
  try {
    require("@webex/internal-plugin-device");
    require("@webex/internal-plugin-mercury");
    require("@webex/plugin-logger");
    require("@webex/plugin-people");
    require("@webex/plugin-rooms");
    require("@webex/plugin-messages");
    const coreModule = require("@webex/webex-core");
    WebexCore = coreModule.default || coreModule;
  } catch (error) {
    emit("fatal", {
      message:
        "The Webex JS SDK packages are not installed. Run `npm install` in the repo root before starting Hermes.",
      detail: formatError(error),
    });
    process.exit(1);
    return;
  }

  const webex = new WebexCore({
    credentials: {
      authorization: token,
    },
    config: {
      credentials: {},
      logger: {
        level: process.env.WEBEX_LOG_LEVEL,
      },
    },
  });

  let shuttingDown = false;

  const shutdown = async (signal) => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    emit("log", { level: "info", message: `Webex listener shutting down (${signal})` });
    try {
      if (typeof webex.messages.stopListening === "function") {
        await webex.messages.stopListening();
      }
    } catch (error) {
      emit("log", {
        level: "warning",
        message: `Webex stopListening failed: ${formatError(error)}`,
      });
    }
    try {
      if (webex.internal && webex.internal.mercury && typeof webex.internal.mercury.disconnect === "function") {
        await webex.internal.mercury.disconnect();
      }
    } catch (error) {
      emit("log", {
        level: "warning",
        message: `Webex mercury disconnect failed: ${formatError(error)}`,
      });
    }
    process.exit(0);
  };

  process.on("SIGINT", () => {
    void shutdown("SIGINT");
  });
  process.on("SIGTERM", () => {
    void shutdown("SIGTERM");
  });
  process.on("uncaughtException", (error) => {
    emit("fatal", { message: formatError(error) });
    process.exit(1);
  });
  process.on("unhandledRejection", (reason) => {
    emit("fatal", { message: formatError(reason) });
    process.exit(1);
  });

  webex.messages.on("created", (event) => {
    emit("event", { event });
  });

  await webex.messages.listen();

  let bot = null;
  try {
    bot = await webex.people.get("me");
  } catch (error) {
    emit("log", {
      level: "warning",
      message: `Webex identity lookup failed after websocket connect: ${formatError(error)}`,
    });
  }

  emit("ready", {
    bot: bot
      ? {
          id: bot.id || "",
          displayName: bot.displayName || "",
          email: Array.isArray(bot.emails) && bot.emails.length > 0 ? bot.emails[0] : "",
        }
      : null,
  });
}

void main().catch((error) => {
  emit("fatal", { message: formatError(error) });
  process.exit(1);
});
