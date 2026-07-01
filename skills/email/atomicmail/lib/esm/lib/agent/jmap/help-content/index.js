// Assembled help topics for MCP `help` and AgentSkill `help`.
import { readNpmPackageReadme } from "../../../core/read-npm-package-readme.js";
import { tryReadSharedJson, tryReadSharedText, } from "../../../core/shared-assets.js";
import { helpTopicAuth } from "./auth.js";
import { helpTopicCron } from "./cron.js";
import { helpTopicInstallation } from "./installation.js";
import { helpTopicJmapCheatsheet } from "./jmap-cheatsheet.js";
import { helpTopicMultiAccount } from "./multi-account.js";
import { helpTopicOverview } from "./overview.js";
import { helpTopicPresets } from "./presets.js";
import { helpTopicTools } from "./tools.js";
import { helpTopicTroubleshooting } from "./troubleshooting.js";
const manifest = tryReadSharedJson("manifest.json");
const errors = tryReadSharedJson("messages/errors.json");
const fallbackTopics = {
    overview: helpTopicOverview,
    installation: helpTopicInstallation,
    auth: helpTopicAuth,
    jmap_cheatsheet: helpTopicJmapCheatsheet,
    tools: helpTopicTools,
    presets: helpTopicPresets,
    cron: helpTopicCron,
    multi_account: helpTopicMultiAccount,
    troubleshooting: helpTopicTroubleshooting,
};
const DEFAULT_README_STUB = 'Topic "readme" returns a built-in stub in AgentSkill runtimes. From MCP, topic "readme" returns the package README.md.';
const DEFAULT_UNKNOWN_TOPIC = "Unknown topic \"{topic}\". Available topics: {topics}, readme";
export const HELP_TOPICS = manifest
    ? Object.fromEntries(manifest.help.topic_order.map((topic) => {
        const text = tryReadSharedText(`${manifest.help.topics_dir}/${topic}.md`) ??
            fallbackTopics[topic];
        return [topic, text];
    }))
    : fallbackTopics;
export const HELP_TOPIC_LIST = manifest
    ? [...manifest.help.topic_order]
    : Object.keys(fallbackTopics);
const HELP_README_STUB = manifest
    ? (tryReadSharedText(manifest.help.readme_stub_path) ?? DEFAULT_README_STUB)
        .trim()
    : DEFAULT_README_STUB;
export function normalizeHelpTopic(topic) {
    return topic.toLowerCase().replace(/[\s-]/g, "_");
}
export async function getHelp(topic, runtime = "skill") {
    if (!topic) {
        return HELP_TOPICS["overview"];
    }
    const key = normalizeHelpTopic(topic);
    if (key === "readme") {
        if (runtime === "mcp") {
            return await readNpmPackageReadme();
        }
        return HELP_README_STUB;
    }
    const unknownTemplate = errors?.help_unknown_topic_template ??
        DEFAULT_UNKNOWN_TOPIC;
    return (HELP_TOPICS[key] ?? unknownTemplate)
        .replace("{topic}", topic)
        .replace("{topics}", HELP_TOPIC_LIST.join(", "));
}
