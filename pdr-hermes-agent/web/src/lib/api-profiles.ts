/** Receipt returned by the dashboard's POST /api/profiles endpoint. */
export interface ProfileCreateResponse {
  ok: boolean;
  name: string;
  path: string;
  model_set?: boolean;
  warnings?: string[];
  mcp_written?: number;
  skills_disabled?: number;
  hub_installs?: Array<{ identifier: string; pid: number | null }>;
}
