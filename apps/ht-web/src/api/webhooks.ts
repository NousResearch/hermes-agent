// Typed wrappers over the webhooks admin REST API. Shapes mirror
// web/src/lib/api.ts (the source of truth); only the subset the WebhooksPage
// consumes is declared.
import { apiGet, apiPost, apiPut, apiDelete } from "./client";

export interface WebhookRoute {
  name: string;
  description: string;
  events: string[];
  deliver: string;
  deliver_only: boolean;
  prompt: string;
  skills: string[];
  created_at: string | null;
  url: string;
  secret_set: boolean;
  enabled: boolean;
}

export interface WebhooksResponse {
  enabled: boolean;
  base_url: string;
  subscriptions: WebhookRoute[];
}

export interface WebhookEnableResponse {
  ok: boolean;
  platform: "webhook";
  enabled: true;
  needs_restart: boolean;
  restart_started?: boolean;
  restart_action?: string;
  restart_pid?: number | null;
  restart_error?: string;
}

export interface WebhookCreate {
  name: string;
  description?: string;
  events?: string[];
  prompt?: string;
  skills?: string[];
  deliver?: string;
  deliver_only?: boolean;
  deliver_chat_id?: string;
}

/** The one-time secret is only returned once, on creation. */
export type WebhookCreated = WebhookRoute & { secret: string };

/** GET /api/webhooks */
export const getWebhooks = () => apiGet<WebhooksResponse>("/api/webhooks");

/** POST /api/webhooks/enable */
export const enableWebhooks = () =>
  apiPost<WebhookEnableResponse>("/api/webhooks/enable");

/** POST /api/webhooks */
export const createWebhook = (body: WebhookCreate) =>
  apiPost<WebhookCreated>("/api/webhooks", body);

/** DELETE /api/webhooks/{name} */
export const deleteWebhook = (name: string) =>
  apiDelete<{ ok: boolean }>(`/api/webhooks/${encodeURIComponent(name)}`);

/** PUT /api/webhooks/{name}/enabled */
export const setWebhookEnabled = (name: string, enabled: boolean) =>
  apiPut<{ ok: boolean; name: string; enabled: boolean }>(
    `/api/webhooks/${encodeURIComponent(name)}/enabled`,
    { enabled },
  );
