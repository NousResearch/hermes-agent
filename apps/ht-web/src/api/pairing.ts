// Typed wrappers over the pairing admin REST API. Shapes mirror
// web/src/lib/api.ts (the source of truth); only the subset the PairingPage
// consumes is declared.
import { apiGet, apiPost } from "./client";

export interface PairingUser {
  platform: string;
  user_id: string;
  user_name?: string;
  code?: string;
  age_minutes?: number;
}

export interface PairingResponse {
  pending: PairingUser[];
  approved: PairingUser[];
}

/** GET /api/pairing */
export const getPairing = () => apiGet<PairingResponse>("/api/pairing");

/** POST /api/pairing/approve */
export const approvePairing = (platform: string, code: string) =>
  apiPost<{ ok: boolean; user: PairingUser }>("/api/pairing/approve", { platform, code });

/** POST /api/pairing/revoke */
export const revokePairing = (platform: string, user_id: string) =>
  apiPost<{ ok: boolean }>("/api/pairing/revoke", { platform, user_id });

/** POST /api/pairing/clear-pending */
export const clearPendingPairing = () =>
  apiPost<{ ok: boolean; cleared: number }>("/api/pairing/clear-pending");
