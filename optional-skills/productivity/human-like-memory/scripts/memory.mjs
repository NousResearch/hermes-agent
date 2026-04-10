#!/usr/bin/env node
/**
 * Human-Like Memory CLI for Hermes Skill
 *
 * Usage:
 *   node memory.mjs recall "query"
 *   node memory.mjs save "user message" "assistant response"
 *   node memory.mjs save-batch                    # reads JSON from stdin
 *   node memory.mjs search "query"
 *   node memory.mjs config
 */

import { createInterface } from 'readline';

const SKILL_VERSION = '1.0.0';

/**
 * Parse boolean from string/boolean value
 */
function parseBoolean(value, defaultValue) {
  if (value === undefined || value === null) return defaultValue;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    return value.toLowerCase() === 'true' || value === '1';
  }
  return defaultValue;
}

/**
 * Build configuration from environment variables
 */
async function buildConfig() {
  const rawLimit = parseInt(process.env.HUMAN_LIKE_MEM_LIMIT_NUMBER || '', 10);
  const rawMinScore = parseFloat(process.env.HUMAN_LIKE_MEM_MIN_SCORE || '');
  const rawTimeoutMs = parseInt(process.env.HUMAN_LIKE_MEM_TIMEOUT_MS || '', 10);
  const rawSaveTriggerTurns = parseInt(process.env.HUMAN_LIKE_MEM_SAVE_TRIGGER_TURNS || '', 10);
  const rawSaveMaxMessages = parseInt(process.env.HUMAN_LIKE_MEM_SAVE_MAX_MESSAGES || '', 10);
  return {
    baseUrl: process.env.HUMAN_LIKE_MEM_BASE_URL || 'https://plugin.human-like.me',
    apiKey: process.env.HUMAN_LIKE_MEM_API_KEY,
    userId: process.env.HUMAN_LIKE_MEM_USER_ID || 'hermes-user',
    agentId: process.env.HUMAN_LIKE_MEM_AGENT_ID || 'main',
    memoryLimitNumber: Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 6,
    minScore: Number.isFinite(rawMinScore) ? rawMinScore : 0.1,
    timeoutMs: Number.isFinite(rawTimeoutMs) && rawTimeoutMs > 0 ? rawTimeoutMs : 30000,
    scenario: process.env.HUMAN_LIKE_MEM_SCENARIO || 'humanlike-memory-skill-smart',
    recallEnabled: parseBoolean(process.env.HUMAN_LIKE_MEM_RECALL_ENABLED, true),
    addEnabled: parseBoolean(process.env.HUMAN_LIKE_MEM_ADD_ENABLED, true),
    autoSaveEnabled: parseBoolean(process.env.HUMAN_LIKE_MEM_AUTO_SAVE_ENABLED, true),
    saveTriggerTurns: Number.isFinite(rawSaveTriggerTurns) && rawSaveTriggerTurns > 0 ? rawSaveTriggerTurns : 5,
    saveMaxMessages: Number.isFinite(rawSaveMaxMessages) && rawSaveMaxMessages > 0 ? rawSaveMaxMessages : 20,
  };
}

/**
 * Build actionable guidance when API key is missing
 */
function buildMissingApiKeyError() {
  return {
    success: false,
    error: 'API Key not configured. HUMAN_LIKE_MEM_API_KEY is required.',
    nextSteps: [
      'Set environment variable: export HUMAN_LIKE_MEM_API_KEY="mp_xxx"',
      'Optional: export HUMAN_LIKE_MEM_BASE_URL="https://plugin.human-like.me"',
      'Optional: export HUMAN_LIKE_MEM_USER_ID="hermes-user"',
      'Then verify with: node ./scripts/memory.mjs config',
    ],
    helpUrl: 'https://plugin.human-like.me',
  };
}

/**
 * Make HTTP request with timeout
 */
async function httpRequest(url, options, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText.substring(0, 200)}`);
    }
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeoutMs}ms`);
    }
    throw error;
  }
}

/**
 * Recall memories based on query
 */
async function recallMemory(query) {
  const cfg = await buildConfig();

  if (!cfg.apiKey) {
    console.error(JSON.stringify(buildMissingApiKeyError()));
    process.exit(1);
  }

  if (!cfg.recallEnabled) {
    console.log(JSON.stringify({
      success: true,
      count: 0,
      memories: [],
      message: 'Memory recall is disabled via HUMAN_LIKE_MEM_RECALL_ENABLED=false',
    }, null, 2));
    return;
  }

  const url = `${cfg.baseUrl}/api/plugin/v1/search/memory`;
  const payload = {
    query: query,
    user_id: cfg.userId,
    agent_id: cfg.agentId,
    memory_limit_number: cfg.memoryLimitNumber,
    min_score: cfg.minScore,
  };

  try {
    const result = await httpRequest(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': cfg.apiKey,
        'x-request-id': `humanlike-memory-skill-${Date.now()}`,
        'x-plugin-version': SKILL_VERSION,
        'x-client-type': 'skill',
      },
      body: JSON.stringify(payload),
    }, cfg.timeoutMs);

    if (!result.success) {
      console.error(JSON.stringify({
        success: false,
        error: result.error || 'Memory retrieval failed',
      }));
      process.exit(1);
    }

    const memories = result.memories || [];

    // Format output for agent consumption
    const output = {
      success: true,
      count: memories.length,
      memories: memories.map(m => ({
        content: m.description || m.event || '',
        timestamp: m.timestamp,
        score: m.score,
      })),
    };

    // Also output human-readable format for context injection
    if (memories.length > 0) {
      output.context = formatMemoriesForContext(memories);
    }

    console.log(JSON.stringify(output, null, 2));
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: error.message,
    }));
    process.exit(1);
  }
}

/**
 * Save messages to memory
 */
async function saveMemory(userMessage, assistantResponse) {
  const cfg = await buildConfig();

  if (!cfg.apiKey) {
    console.error(JSON.stringify(buildMissingApiKeyError()));
    process.exit(1);
  }

  if (!cfg.addEnabled) {
    console.log(JSON.stringify({
      success: true,
      message: 'Memory storage is disabled via HUMAN_LIKE_MEM_ADD_ENABLED=false',
    }));
    return;
  }

  const url = `${cfg.baseUrl}/api/plugin/v1/add/message`;
  const sessionId = `session-${Date.now()}`;
  const agentId = cfg.agentId || 'main';
  const messages = [];

  if (userMessage) {
    messages.push({ role: 'user', content: userMessage });
  }
  if (assistantResponse) {
    messages.push({ role: 'assistant', content: assistantResponse });
  }

  if (messages.length === 0) {
    console.error(JSON.stringify({
      success: false,
      error: 'No messages to save',
    }));
    process.exit(1);
  }

  const payload = {
    user_id: cfg.userId,
    conversation_id: sessionId,
    messages: messages,
    agent_id: agentId,
    tags: ['humanlike-memory-skill'],
    async_mode: true,
    custom_workflows: {
      stream_params: {
        metadata: JSON.stringify({
          user_ids: [cfg.userId],
          agent_ids: [agentId],
          session_id: sessionId,
          scenario: cfg.scenario || 'humanlike-memory-skill',
        }),
      },
    },
  };

  try {
    const result = await httpRequest(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': cfg.apiKey,
        'x-request-id': `humanlike-memory-skill-${Date.now()}`,
        'x-plugin-version': SKILL_VERSION,
        'x-client-type': 'skill',
      },
      body: JSON.stringify(payload),
    }, cfg.timeoutMs);

    const output = {
      success: true,
      message: 'Memory saved successfully',
      memoriesCount: result.memories_count || 0,
    };

    console.log(JSON.stringify(output));
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: error.message,
    }));
    process.exit(1);
  }
}

/**
 * Search memories (alias for recall with different output format)
 */
async function searchMemory(query) {
  await recallMemory(query);
}

/**
 * Read JSON from stdin
 */
async function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';
    const rl = createInterface({
      input: process.stdin,
      terminal: false,
    });

    rl.on('line', (line) => {
      data += line;
    });

    rl.on('close', () => {
      resolve(data.trim());
    });

    rl.on('error', reject);

    // Timeout after 5 seconds if no input
    setTimeout(() => {
      rl.close();
      if (!data) {
        reject(new Error('No input received from stdin'));
      }
    }, 5000);
  });
}

/**
 * Save batch messages to memory (from stdin JSON)
 */
async function saveBatchMemory() {
  const cfg = await buildConfig();

  if (!cfg.apiKey) {
    console.error(JSON.stringify(buildMissingApiKeyError()));
    process.exit(1);
  }

  if (!cfg.addEnabled) {
    console.log(JSON.stringify({
      success: true,
      message: 'Memory storage is disabled via HUMAN_LIKE_MEM_ADD_ENABLED=false',
    }));
    return;
  }

  let inputData;
  try {
    inputData = await readStdin();
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: `Failed to read stdin: ${error.message}`,
      usage: 'echo \'[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]\' | node memory.mjs save-batch',
    }));
    process.exit(1);
  }

  let messages;
  try {
    messages = JSON.parse(inputData);
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: `Invalid JSON: ${error.message}`,
      received: inputData.substring(0, 200),
    }));
    process.exit(1);
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    console.error(JSON.stringify({
      success: false,
      error: 'Messages must be a non-empty array',
    }));
    process.exit(1);
  }

  // Validate message format
  for (const msg of messages) {
    if (!msg.role || !msg.content) {
      console.error(JSON.stringify({
        success: false,
        error: 'Each message must have "role" and "content" fields',
        invalid: msg,
      }));
      process.exit(1);
    }
    if (!['user', 'assistant'].includes(msg.role)) {
      console.error(JSON.stringify({
        success: false,
        error: 'Role must be "user" or "assistant"',
        invalid: msg.role,
      }));
      process.exit(1);
    }
  }

  const maxMessages = cfg.saveMaxMessages;
  const messagesToSave = messages.slice(-maxMessages);

  const url = `${cfg.baseUrl}/api/plugin/v1/add/message`;
  const sessionId = `session-${Date.now()}`;
  const agentId = cfg.agentId || 'main';

  const payload = {
    user_id: cfg.userId,
    conversation_id: sessionId,
    messages: messagesToSave.map(m => ({
      role: m.role,
      content: m.content.substring(0, 20000), // Truncate if too long
    })),
    agent_id: agentId,
    tags: ['humanlike-memory-skill'],
    async_mode: true,
    custom_workflows: {
      stream_params: {
        metadata: JSON.stringify({
          user_ids: [cfg.userId],
          agent_ids: [agentId],
          session_id: sessionId,
          scenario: cfg.scenario || 'humanlike-memory-skill',
        }),
      },
    },
  };

  try {
    const result = await httpRequest(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': cfg.apiKey,
        'x-request-id': `humanlike-memory-skill-${Date.now()}`,
        'x-plugin-version': SKILL_VERSION,
        'x-client-type': 'skill',
      },
      body: JSON.stringify(payload),
    }, cfg.timeoutMs);

    const turnCount = Math.floor(messagesToSave.length / 2);
    const output = {
      success: true,
      message: `Saved ${turnCount} turns (${messagesToSave.length} messages) to memory`,
      memoriesCount: result.memories_count || 0,
      config: {
        saveMaxMessages: cfg.saveMaxMessages,
      },
    };

    console.log(JSON.stringify(output));
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: error.message,
    }));
    process.exit(1);
  }
}

/**
 * Show current configuration (without sensitive data)
 */
async function showConfig() {
  const cfg = await buildConfig();

  console.log(JSON.stringify({
    baseUrl: cfg.baseUrl,
    userId: cfg.userId,
    agentId: cfg.agentId,
    apiKeyConfigured: !!cfg.apiKey,
    memoryLimitNumber: cfg.memoryLimitNumber,
    minScore: cfg.minScore,
    recallEnabled: cfg.recallEnabled,
    addEnabled: cfg.addEnabled,
    autoSaveEnabled: cfg.autoSaveEnabled,
    saveTriggerTurns: cfg.saveTriggerTurns,
    saveMaxMessages: cfg.saveMaxMessages,
    mode: 'agent-smart',
  }, null, 2));
}

/**
 * Format memories for context injection (aligned with plugin format)
 */
function formatMemoriesForContext(memories) {
  if (!memories || memories.length === 0) return '';

  const now = Date.now();
  const nowText = formatTime(now);

  const memoryLines = memories
    .map(m => {
      const date = formatTime(m.timestamp);
      const content = m.description || m.event || '';
      const score = m.score ? ` (${(m.score * 100).toFixed(0)}%)` : '';
      if (!content) return '';
      if (date) return `   -[${date}] ${content}${score}`;
      return `   - ${content}${score}`;
    })
    .filter(Boolean);

  if (memoryLines.length === 0) return '';

  const lines = [
    '# Role',
    '',
    'You are an intelligent assistant with long-term memory capabilities. Your goal is to combine retrieved memory fragments to provide highly personalized, accurate, and logically rigorous responses.',
    '',
    '# System Context',
    '',
    `* Current Time: ${nowText} (Use this as the baseline for freshness checks)`,
    '',
    '# Memory Data',
    '',
    'Below are **episodic memory summaries** retrieved from long-term memory.',
    '',
    '* **Memory Type**: All memories are episodic summaries - they represent contextual information from past conversations.',
    '* **Special Note**: If content is tagged with \'[assistant_opinion]\' or \'[model_summary]\', it represents **past AI inference**, **not** direct user statements.',
    '',
    '```text',
    '<memories>',
    ...memoryLines,
    '</memories>',
    '```',
    '',
    '# Critical Protocol: Memory Safety',
    '',
    '1. **Source Verification**: Distinguish direct user statements from AI inference. AI summaries are reference-only.',
    '2. **Attribution Check**: Never attribute third-party info to the user.',
    '3. **Strong Relevance Check**: Only use memories that directly help answer the current query.',
    '4. **Freshness Check**: Prioritize the current query over conflicting memories.',
    '',
    '# Instructions',
    '',
    '1. **Review**: Read the episodic memory summaries and apply the protocol above to remove noise.',
    '2. **Execute**: Use only memories that pass filtering as context.',
    '3. **Output**: Answer directly. Never mention internal terms such as "memory store", "retrieval", or "AI opinions".',
  ];

  return lines.join('\n');
}

/**
 * Format timestamp for display
 */
function formatTime(value) {
  if (!value) return '';
  if (typeof value === 'number') {
    const date = new Date(value);
    if (isNaN(date.getTime())) return '';
    const pad = (v) => String(v).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  }
  return String(value);
}

/**
 * Print usage information
 */
function printUsage() {
  console.log(`
Human-Like Memory CLI for Hermes

Usage:
  node memory.mjs <command> [arguments]

Commands:
  recall <query>                    Retrieve relevant memories for a query
  save <user_msg> [assistant_msg]   Save single conversation turn to memory
  save-batch                        Save multiple turns from stdin (JSON array)
  search <query>                    Search memories (alias for recall)
  config                            Show current configuration

Examples:
  node memory.mjs recall "What projects am I working on?"
  node memory.mjs save "I'm working on Project X" "Great, I'll remember that."
  node memory.mjs search "meeting notes"
  node memory.mjs config

  # Save batch (read JSON array from file):
  node memory.mjs save-batch < ./examples/messages.json

Configuration:
  Configure via Hermes secure env or environment variables:
  - HUMAN_LIKE_MEM_API_KEY (required)
  - HUMAN_LIKE_MEM_BASE_URL (optional, default: https://plugin.human-like.me)
  - HUMAN_LIKE_MEM_USER_ID (optional, default: hermes-user)
  - HUMAN_LIKE_MEM_AGENT_ID (optional, default: main)
  - HUMAN_LIKE_MEM_LIMIT_NUMBER (optional, default: 6)
  - HUMAN_LIKE_MEM_MIN_SCORE (optional, default: 0.1)
  - HUMAN_LIKE_MEM_RECALL_ENABLED (optional, default: true)
  - HUMAN_LIKE_MEM_ADD_ENABLED (optional, default: true)
  - HUMAN_LIKE_MEM_AUTO_SAVE_ENABLED (optional, default: true)
  - HUMAN_LIKE_MEM_SAVE_TRIGGER_TURNS (optional, default: 5)
  - HUMAN_LIKE_MEM_SAVE_MAX_MESSAGES (optional, default: 20)
`);
}

// Main entry point
const [,, command, ...args] = process.argv;

switch (command) {
  case 'recall':
    if (!args[0]) {
      console.error('Error: Query is required for recall command');
      process.exit(1);
    }
    await recallMemory(args.join(' '));
    break;

  case 'save':
    if (!args[0]) {
      console.error('Error: At least one message is required for save command');
      process.exit(1);
    }
    await saveMemory(args[0], args[1]);
    break;

  case 'save-batch':
    await saveBatchMemory();
    break;

  case 'search':
    if (!args[0]) {
      console.error('Error: Query is required for search command');
      process.exit(1);
    }
    await searchMemory(args.join(' '));
    break;

  case 'config':
    await showConfig();
    break;

  case 'help':
  case '--help':
  case '-h':
    printUsage();
    break;

  default:
    if (command) {
      console.error(`Unknown command: ${command}`);
    }
    printUsage();
    process.exit(1);
}
