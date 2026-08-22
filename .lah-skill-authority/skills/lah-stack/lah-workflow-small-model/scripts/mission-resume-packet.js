#!/usr/bin/env node
/**
 * Hermes Mission Resume Packet v1
 *
 * Persistent mission resume packets for resuming missions from a known checkpoint.
 * Eliminates cold-start rediscovery by loading the exact state needed to continue.
 *
 * Implements:
 *   P4 — Mission Resume Packet (persistent, loadable)
 *   P8 — Resume Directly (checkpoint honored, no architecture inspection)
 *
 * @module mission-resume-packet
 * @version 1.0.0
 */

"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

// ─── Constants ───────────────────────────────────────────────────

const RESUME_PACKET_PATH = path.join(
  __dirname,
  "..",
  "data",
  "mission-resume-packet.json"
);

const PACKET_VERSION = "1.0.0";

// ─── Required Fields (P4) ──────────────────────────────────────

const REQUIRED_FIELDS = Object.freeze([
  "mission_id",
  "mission_type",
  "current_checkpoint",
  "next_action",
  "known_facts",
  "resolved_blockers",
  "blocking_unknowns",
  "authorization_state",
  "provider_state",
  "approval_ids",
  "campaign_ids",
  "compiled_packet_ids",
  "canonical_context_fingerprint",
  "forbidden_rediscovery",
  "last_verified_at",
]);

// ─── Resume Packet Class ──────────────────────────────────────

class MissionResumePacket {
  constructor(options = {}) {
    this.packetPath = options.packetPath || RESUME_PACKET_PATH;
    this.packet = null;
  }

  /**
   * Create a new resume packet from mission state.
   *
   * @param {object} params - Packet parameters
   * @param {string} params.mission_id - Unique mission identifier
   * @param {string} params.mission_type - Mission type (EXECUTE, DIAGNOSTIC, REPAIR, CERTIFY, BUILD_AND_CERTIFY)
   * @param {string} params.current_checkpoint - Current checkpoint (e.g., PENDING_OPERATOR_APPROVAL)
   * @param {string} params.next_action - The exact next action to take
   * @param {object} params.known_facts - Map of fact_key → certified value
   * @param {string[]} params.resolved_blockers - Blockers that have been resolved
   * @param {string[]} params.blocking_unknowns - Unknowns still blocking progress
   * @param {object} params.authorization_state - Current authorization state
   * @param {object} params.provider_state - Provider state (campaign IDs, etc.)
   * @param {string[]} params.approval_ids - Approval IDs in the pipeline
   * @param {string[]} params.campaign_ids - Campaign IDs in the pipeline
   * @param {string[]} params.compiled_packet_ids - Compiled packet IDs
   * @param {string} params.canonical_context_fingerprint - LAH_ARCHITECTURE_FINGERPRINT
   * @param {string[]} params.forbidden_rediscovery - Facts that must NOT be rediscovered
   * @returns {object} The created packet
   */
  create(params) {
    // Validate required fields
    const missing = REQUIRED_FIELDS.filter((f) => !(f in params));
    if (missing.length > 0) {
      throw new Error(
        `RESUME_PACKET_INCOMPLETE: Missing required fields: ${missing.join(", ")}`
      );
    }

    const packet = {
      version: PACKET_VERSION,
      mission_id: params.mission_id,
      mission_type: params.mission_type,
      current_checkpoint: params.current_checkpoint,
      next_action: params.next_action,
      known_facts: params.known_facts,
      resolved_blockers: params.resolved_blockers || [],
      blocking_unknowns: params.blocking_unknowns || [],
      authorization_state: params.authorization_state,
      provider_state: params.provider_state || {},
      approval_ids: params.approval_ids || [],
      campaign_ids: params.campaign_ids || [],
      compiled_packet_ids: params.compiled_packet_ids || [],
      canonical_context_fingerprint: params.canonical_context_fingerprint,
      forbidden_rediscovery: params.forbidden_rediscovery || [],
      last_verified_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      packet_id: crypto
        .createHash("sha256")
        .update(`${params.mission_id}:${Date.now()}`)
        .digest("hex"),
    };

    this.packet = packet;
    this.persist();

    return packet;
  }

  /**
   * P8: Load a resume packet from persistent storage.
   * A mission continuation MUST load this packet before any discovery.
   *
   * @param {string} [missionId] - Optional mission ID to filter by
   * @returns {{ success: boolean, packet?: object, reason?: string }}
   */
  load(missionId) {
    const startTime = Date.now();

    if (!fs.existsSync(this.packetPath)) {
      return {
        success: false,
        reason: "NO_RESUME_PACKET: No resume packet found on disk",
        load_ms: Date.now() - startTime,
      };
    }

    try {
      const data = JSON.parse(fs.readFileSync(this.packetPath, "utf8"));

      // Validate required fields
      const missing = REQUIRED_FIELDS.filter((f) => !(f in data));
      if (missing.length > 0) {
        return {
          success: false,
          reason: `RESUME_PACKET_CORRUPT: Missing required fields: ${missing.join(", ")}`,
          load_ms: Date.now() - startTime,
        };
      }

      // If missionId specified, verify it matches
      if (missionId && data.mission_id !== missionId) {
        return {
          success: false,
          reason: `RESUME_PACKET_MISMATCH: Packet is for mission '${data.mission_id}', requested '${missionId}'`,
          load_ms: Date.now() - startTime,
        };
      }

      this.packet = data;

      return {
        success: true,
        packet: data,
        load_ms: Date.now() - startTime,
      };
    } catch (e) {
      return {
        success: false,
        reason: `RESUME_PACKET_PARSE_ERROR: ${e.message}`,
        load_ms: Date.now() - startTime,
      };
    }
  }

  /**
   * Get the current checkpoint.
   * @returns {string|null}
   */
  getCurrentCheckpoint() {
    return this.packet ? this.packet.current_checkpoint : null;
  }

  /**
   * Get the next action.
   * @returns {string|null}
   */
  getNextAction() {
    return this.packet ? this.packet.next_action : null;
  }

  /**
   * P8: Get the exact approval ID for direct execution.
   * @returns {string|null}
   */
  getApprovalId() {
    if (!this.packet || !this.packet.approval_ids || this.packet.approval_ids.length === 0) {
      return null;
    }
    return this.packet.approval_ids[0];
  }

  /**
   * Check if the packet is valid for resuming.
   * @returns {{ valid: boolean, issues: string[] }}
   */
  validate() {
    const issues = [];

    if (!this.packet) {
      return { valid: false, issues: ["No packet loaded"] };
    }

    // Check required fields
    const missing = REQUIRED_FIELDS.filter((f) => !(f in this.packet));
    if (missing.length > 0) {
      issues.push(`Missing required fields: ${missing.join(", ")}`);
    }

    // Check fingerprint match
    if (this.packet.canonical_context_fingerprint) {
      // Fingerprint will be verified externally
    }

    // Check if checkpoint is terminal
    const terminalCheckpoints = ["COMPLETED", "FAILED", "CANCELLED"];
    if (terminalCheckpoints.includes(this.packet.current_checkpoint)) {
      issues.push(`Checkpoint is terminal: ${this.packet.current_checkpoint}`);
    }

    return {
      valid: issues.length === 0,
      issues,
    };
  }

  /**
   * Update the packet's checkpoint and next action.
   * @param {string} newCheckpoint - The new checkpoint
   * @param {string} newNextAction - The new next action
   */
  advance(newCheckpoint, newNextAction) {
    if (!this.packet) {
      throw new Error("No packet loaded — call load() first");
    }

    this.packet.current_checkpoint = newCheckpoint;
    this.packet.next_action = newNextAction;
    this.packet.updated_at = new Date().toISOString();

    this.persist();

    return {
      advanced: true,
      from_checkpoint: newCheckpoint,
      to_next_action: newNextAction,
    };
  }

  /**
   * Add an approval ID to the packet.
   * @param {string} approvalId
   */
  addApprovalId(approvalId) {
    if (!this.packet) return;
    if (!this.packet.approval_ids.includes(approvalId)) {
      this.packet.approval_ids.push(approvalId);
      this.packet.updated_at = new Date().toISOString();
      this.persist();
    }
  }

  /**
   * Add a campaign ID to the packet.
   * @param {string} campaignId
   */
  addCampaignId(campaignId) {
    if (!this.packet) return;
    if (!this.packet.campaign_ids.includes(campaignId)) {
      this.packet.campaign_ids.push(campaignId);
      this.packet.updated_at = new Date().toISOString();
      this.persist();
    }
  }

  /**
   * Add a compiled packet ID to the packet.
   * @param {string} packetId
   */
  addCompiledPacketId(packetId) {
    if (!this.packet) return;
    if (!this.packet.compiled_packet_ids.includes(packetId)) {
      this.packet.compiled_packet_ids.push(packetId);
      this.packet.updated_at = new Date().toISOString();
      this.persist();
    }
  }

  /**
   * Add a resolved blocker.
   * @param {string} blocker
   */
  resolveBlocker(blocker) {
    if (!this.packet) return;
    const idx = this.packet.resolved_blockers.indexOf(blocker);
    if (idx === -1) {
      this.packet.resolved_blockers.push(blocker);
      const bIdx = this.packet.blocking_unknowns.indexOf(blocker);
      if (bIdx !== -1) {
        this.packet.blocking_unknowns.splice(bIdx, 1);
      }
      this.packet.updated_at = new Date().toISOString();
      this.persist();
    }
  }

  /**
   * Add a blocking unknown.
   * @param {string} unknown
   */
  addBlockingUnknown(unknown) {
    if (!this.packet) return;
    if (!this.packet.blocking_unknowns.includes(unknown)) {
      this.packet.blocking_unknowns.push(unknown);
      this.packet.updated_at = new Date().toISOString();
      this.persist();
    }
  }

  /**
   * Persist the packet to disk.
   */
  persist() {
    if (!this.packet) return;
    const dir = path.dirname(this.packetPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(this.packetPath, JSON.stringify(this.packet, null, 2));
  }

  /**
   * Get the raw packet.
   * @returns {object|null}
   */
  getPacket() {
    return this.packet ? { ...this.packet } : null;
  }

  /**
   * Delete the resume packet (for cleanup).
   */
  delete() {
    if (fs.existsSync(this.packetPath)) {
      fs.unlinkSync(this.packetPath);
    }
    this.packet = null;
  }

  /**
   * Check if a resume packet exists.
   * @returns {boolean}
   */
  exists() {
    return fs.existsSync(this.packetPath);
  }
}

// ─── Export ──────────────────────────────────────────────────────

module.exports = {
  MissionResumePacket,
  REQUIRED_FIELDS,
  PACKET_VERSION,
  RESUME_PACKET_PATH,
};