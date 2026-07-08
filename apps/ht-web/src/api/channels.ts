// Typed wrappers for the messaging-platforms API (/api/messaging/*). Shapes
// mirror web/src/lib/api.ts (the source of truth); only the subset the Channels
// page consumes is declared here.
//
// NOTE: the Telegram/WhatsApp guided onboarding wizards
// (POST /api/messaging/{telegram,whatsapp}/onboarding/*) are intentionally NOT
// ported yet — they are multi-step flows that require polling a session/state
// endpoint between prompts. Deferred; the Channels page only does direct
// enable/config/test here.
import { apiGet, apiPut, apiPost } from "./client";

export interface MessagingPlatformEnvVar {
  key: string;
  required: boolean;
  is_set: boolean;
  redacted_value: string | null;
  description: string;
  prompt: string;
  help: string;
  url: string | null;
  is_password: boolean;
  advanced: boolean;
}

export interface MessagingPlatform {
  id: string;
  name: string;
  description: string;
  docs_url: string;
  enabled: boolean;
  configured: boolean;
  gateway_running: boolean;
  /**
   * "connected" | "disabled" | "not_configured" | "pending_restart" |
   * "gateway_stopped" | "startup_failed" | "disconnected" | "fatal" | string
   */
  state: string;
  error_code: string | null;
  error_message: string | null;
  updated_at: string | null;
  home_channel: {
    platform: string;
    chat_id: string;
    name: string;
    thread_id?: string;
  } | null;
  whatsapp_setup?: {
    mode?: string;
    allowed_users_set?: boolean;
    home_channel_set?: boolean;
  } | null;
  env_vars: MessagingPlatformEnvVar[];
}

export interface MessagingPlatformsResponse {
  env_path: string;
  gateway_start_command: string;
  platforms: MessagingPlatform[];
}

export interface MessagingPlatformUpdate {
  enabled?: boolean;
  env?: Record<string, string>;
  clear_env?: string[];
}

export interface MessagingPlatformTestResult {
  ok: boolean;
  state: string;
  message: string;
}

export const getMessagingPlatforms = () =>
  apiGet<MessagingPlatformsResponse>("/api/messaging/platforms");

export const updateMessagingPlatform = (id: string, body: MessagingPlatformUpdate) =>
  apiPut<{ ok: boolean; platform: string }>(
    `/api/messaging/platforms/${encodeURIComponent(id)}`,
    body,
  );

export const testMessagingPlatform = (id: string) =>
  apiPost<MessagingPlatformTestResult>(
    `/api/messaging/platforms/${encodeURIComponent(id)}/test`,
  );
