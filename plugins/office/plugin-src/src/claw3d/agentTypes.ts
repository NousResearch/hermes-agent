/**
 * Claw3D agent model props — ported from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 *
 * Simplified for Hermes: removed janitor/ping-pong/gym interaction callbacks.
 */

import type { AgentAvatarProfile } from "./profile";
import type { RefObject } from "react";
import type { OfficeAgent, RenderAgent } from "./types";

export type AgentModelProps = {
  agentId: string;
  name: string;
  subtitle?: string | null;
  status: OfficeAgent["status"];
  color: string;
  appearance?: AgentAvatarProfile | null;
  agentsRef: RefObject<RenderAgent[]>;
  agentLookupRef?: RefObject<Map<string, RenderAgent>>;
  onHover?: (id: string) => void;
  onUnhover?: () => void;
  onClick?: (id: string) => void;
  showSpeech?: boolean;
  speechText?: string | null;
  suppressSpeechBubble?: boolean;
};
