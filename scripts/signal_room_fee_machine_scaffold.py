#!/usr/bin/env python3
"""Create a review-only Signal Room fee-machine V2 animation scaffold."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REQUIRED_POSES = (
    "neutral_read",
    "bill_shock",
    "look_to_machine",
    "skeptical_point",
    "slight_lean",
)

MOTION_PRIMITIVES: tuple[dict[str, Any], ...] = (
    {
        "id": "continuous_drift",
        "purpose": "Keep the full stage alive with a slow documentary camera drift.",
        "timeline_marker": "SignalRoomMotion.continuousDrift(",
        "start": 0.0,
        "end": 15.0,
        "readability_role": "prevents frozen-card reads without moving labels out of safe zones",
    },
    {
        "id": "ambient_loop",
        "purpose": "Run subtle scan and grid motion behind the main action.",
        "timeline_marker": "SignalRoomMotion.ambientLoop(",
        "start": 0.0,
        "end": 15.0,
        "readability_role": "adds life while staying secondary to the bill and fee labels",
    },
    {
        "id": "character_micro_acting",
        "purpose": "Layer anticipation, reaction, pointing, and settle motion over pose swaps.",
        "timeline_marker": "SignalRoomMotion.microActing(",
        "start": 0.2,
        "end": 14.7,
        "readability_role": "makes character acting readable as performance at phone size",
    },
    {
        "id": "pose_transition",
        "purpose": "Stage pose swaps with anticipation, readable hold, and settle instead of hard cuts.",
        "timeline_marker": "SignalRoomMotion.poseTransition(",
        "start": 2.35,
        "end": 12.15,
        "readability_role": "turns character pose changes into intentional acting beats",
    },
    {
        "id": "fee_stack_simmer",
        "purpose": "Keep fee tags moving as the machine drives the cost stack.",
        "timeline_marker": "SignalRoomMotion.feeStackSimmer(",
        "start": 3.0,
        "end": 14.5,
        "readability_role": "connects every fee label to the mechanism instead of a static list",
    },
    {
        "id": "machine_idle",
        "purpose": "Give gears, nodes, body, and flow line linked secondary motion.",
        "timeline_marker": "SignalRoomMotion.machineIdle(",
        "start": 5.8,
        "end": 14.7,
        "readability_role": "keeps cause-and-effect visible through reveal, acting beat, and settle",
    },
    {
        "id": "machine_causality",
        "purpose": "Couple lever pulls, gear turns, pulse nodes, and fee-tag pushes into one cause/effect beat.",
        "timeline_marker": "SignalRoomMotion.machineCausality(",
        "start": 6.05,
        "end": 13.3,
        "readability_role": "makes the mechanism visibly drive the fee stack instead of moving as separate parts",
    },
)

SCENE_CHOREOGRAPHY: tuple[dict[str, Any], ...] = (
    {
        "id": "ordinary_bill_hold",
        "start": 0.0,
        "end": 2.5,
        "acting_objective": "read bill before reacting",
        "primary_motion": "quiet camera drift and paper entrance",
        "overlap_in_seconds": 0.0,
        "review_frame": "ordinary_bill",
    },
    {
        "id": "fee_split_reaction",
        "start": 2.35,
        "end": 5.1,
        "acting_objective": "shock reads before the fee labels finish",
        "primary_motion": "fee tags push into view with synced ticks",
        "acting_phases": ["anticipation", "reaction_hold", "settle"],
        "overlap_in_seconds": 0.15,
        "review_frame": "number_split",
    },
    {
        "id": "machine_reveal_handoff",
        "start": 5.0,
        "end": 8.65,
        "acting_objective": "look from bill to the revealed machine",
        "primary_motion": "wall slide reveals active gears and lever",
        "acting_phases": ["anticipation", "look_hold", "settle"],
        "overlap_in_seconds": 0.15,
        "review_frame": "machine_reveal",
    },
    {
        "id": "skeptical_point_drive",
        "start": 8.5,
        "end": 12.1,
        "acting_objective": "point with skepticism at fee source",
        "primary_motion": "pointing pose locks machine cause to fee stack effect",
        "acting_phases": ["anticipation", "point_hold", "settle"],
        "overlap_in_seconds": 0.15,
        "review_frame": "acting_read",
    },
    {
        "id": "memory_anchor_settle",
        "start": 12.0,
        "end": 15.0,
        "acting_objective": "settle into final memory frame",
        "primary_motion": "lever pulse resolves fee stack and caption",
        "acting_phases": ["anticipation", "memory_hold", "settle"],
        "overlap_in_seconds": 0.15,
        "review_frame": "memory_anchor",
    },
)


def require_poses(rig_dir: Path) -> list[Path]:
    poses = []
    for pose in REQUIRED_POSES:
        path = rig_dir / "poses" / f"{pose}.svg"
        if not path.exists():
            raise FileNotFoundError(path)
        poses.append(path)
    return poses


def write_design(out_dir: Path) -> None:
    (out_dir / "DESIGN.md").write_text(
        """# Signal Room Fee Machine V2 Scaffold Design

Status: review-only assembly scaffold.

Palette:
- background: #101820
- panel: #17232d
- ink: #f3eadb
- muted: #8fa0aa
- accent: #d89a3a
- warning: #c85f3f

Typography:
- Inter or Arial fallback.
- Heavy condensed-feeling labels for bill totals.
- Small technical labels only when they clarify mechanism flow.

Direction:
- Adult animated/vector documentary.
- Serious, practical, investigative.
- No childish bounce, mascot energy, or static-card long-form.
"""
    )


def write_expanded_prompt(out_dir: Path) -> None:
    prompt_dir = out_dir / ".hyperframes"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "expanded-prompt.md").write_text(
        """# Expanded Prompt: Fee Machine V2 Review Scaffold

Rhythm: hold -> split -> reveal -> mechanical pulse -> acting beat -> settle

Global rules:
- 1080x1920 vertical review composition.
- Character layer is foreground-left and swaps through five approved SVG poses.
- Bill/fee labels are readable in phone safe zones.
- The mechanism must show cause/effect: lever drives gears, gears push fee tags.
- This is review-only and must not be published.

Beat 1, 0.0-2.5s:
Human reads an ordinary bill. The world is quiet and grounded.

Beat 2, 2.5-5.0s:
The total splits into base price, service fee, processing fee, and platform fee.

Beat 3, 5.0-8.5s:
The wall panel opens and reveals the compact fee machine behind the bill.

Beat 4, 8.5-12.0s:
The character points while the machine visibly drives the fee stack.

Beat 5, 12.0-15.0s:
The lever pulls once more; the fee tags pulse and settle into a memory frame.
"""
    )


def write_audio_cue_sheet(out_dir: Path) -> None:
    cue_sheet = {
        "title": "Signal Room Fee Machine V2 Sound Design Cue Sheet",
        "status": "review-only",
        "duration_seconds": 15,
        "mix_targets": {
            "phone_speaker": True,
            "dialogue_safe": True,
            "music_bed": "low room-tone pulse, no narrator masking",
        },
        "cues": [
            {
                "id": "room_tone_bed",
                "start": 0.0,
                "duration": 15.0,
                "sound": "low office room tone with restrained investigative pulse",
                "mix_note": "keep under all mechanism and narration elements",
            },
            {
                "id": "paper_bill_snap",
                "start": 0.35,
                "duration": 0.25,
                "sound": "dry paper/envelope snap as bill lands",
                "mix_note": "sharp transient, readable on phone speakers",
            },
            {
                "id": "fee_tick_base",
                "start": 2.75,
                "duration": 0.16,
                "sound": "small mechanical tick for base price tag",
                "mix_note": "first tick is slightly lower pitch",
            },
            {
                "id": "fee_tick_service",
                "start": 3.25,
                "duration": 0.16,
                "sound": "small mechanical tick for service fee tag",
                "mix_note": "slightly brighter than base tick",
            },
            {
                "id": "fee_tick_processing",
                "start": 3.75,
                "duration": 0.16,
                "sound": "small mechanical tick for processing fee tag",
                "mix_note": "add faint metal click layer",
            },
            {
                "id": "fee_tick_platform",
                "start": 4.25,
                "duration": 0.18,
                "sound": "largest fee-tag tick with short low tap",
                "mix_note": "marks completion of split",
            },
            {
                "id": "wall_reveal_thump",
                "start": 5.05,
                "duration": 0.75,
                "sound": "damped wall slide plus low mechanical thump",
                "mix_note": "largest reveal sound, no cartoon whoosh",
            },
            {
                "id": "machine_drive_loop",
                "start": 6.05,
                "duration": 3.2,
                "sound": "restrained gear/belt rhythm synced to machine drive",
                "mix_note": "texture only; do not compete with fee labels",
            },
            {
                "id": "pointing_accent",
                "start": 8.5,
                "duration": 0.2,
                "sound": "small cloth/gesture accent as character points",
                "mix_note": "subtle acting support, not slapstick",
            },
            {
                "id": "final_lever_click",
                "start": 12.2,
                "duration": 0.35,
                "sound": "lever pull, latch click, short low impact",
                "mix_note": "anchors final memory-frame pulse",
            },
            {
                "id": "memory_hold_tail",
                "start": 12.6,
                "duration": 2.4,
                "sound": "room tone settles with faint machine decay",
                "mix_note": "leave space for end caption or narration pickup",
            },
        ],
    }
    (out_dir / "audio_cue_sheet.json").write_text(json.dumps(cue_sheet, indent=2) + "\n")


def write_retention_frame_plan(out_dir: Path) -> None:
    frame_plan = {
        "title": "Signal Room Fee Machine V2 Retention Frame Plan",
        "status": "review-only",
        "contact_sheet_required": True,
        "failure_conditions": [
            "five similar cards",
            "human problem is unclear in the first frame",
            "machine reveal does not show cause and effect",
            "character acting read is not visible at phone size",
            "memory anchor depends on caption text alone",
        ],
        "frames": [
            {
                "id": "ordinary_bill",
                "sample_time": 1.25,
                "beat": "human problem",
                "review_question": "Who is affected, and what object starts the story?",
                "must_show": ["foreground character", "ordinary bill", "quiet grounded scene"],
            },
            {
                "id": "number_split",
                "sample_time": 3.75,
                "beat": "visible contradiction",
                "review_question": "Does the simple total visibly split into separate fees?",
                "must_show": ["base price", "service fee", "processing fee", "platform fee"],
            },
            {
                "id": "machine_reveal",
                "sample_time": 6.5,
                "beat": "mechanism reveal",
                "review_question": "Is the fee machine spatially revealed behind the bill?",
                "must_show": ["open wall", "lever or gear driver", "money flow direction"],
            },
            {
                "id": "acting_read",
                "sample_time": 9.25,
                "beat": "escalation",
                "review_question": "Does the character gesture clarify what the mechanism is doing?",
                "must_show": ["skeptical point", "active fee stack", "machine motion"],
            },
            {
                "id": "memory_anchor",
                "sample_time": 13.2,
                "beat": "memory anchor",
                "review_question": "Can this frame survive as the remembered visual for the segment?",
                "must_show": ["final lever pulse", "settled fee stack", "readable caption or visual thesis"],
            },
        ],
    }
    (out_dir / "retention_frame_plan.json").write_text(json.dumps(frame_plan, indent=2) + "\n")


def write_motion_primitives(out_dir: Path) -> None:
    registry = {
        "title": "Signal Room Fee Machine V2 Motion Primitive Registry",
        "status": "review-only",
        "public_release": False,
        "composition_id": "fee-machine-v2-review",
        "primitives": list(MOTION_PRIMITIVES),
    }
    (out_dir / "motion_primitives.json").write_text(json.dumps(registry, indent=2) + "\n")


def write_motion_library(out_dir: Path) -> None:
    (out_dir / "motion_primitives.js").write_text(
        """// Reusable Signal Room fee-machine motion primitives. Review-only.
(function () {
  function cycleRepeat(duration, cycleDuration) {
    return Math.max(1, Math.floor(duration / cycleDuration));
  }

  window.SignalRoomMotion = {
    continuousDrift(tl) {
      tl.fromTo(".stage", { x: -10, y: 4, scale: 1.012 }, { x: 12, y: -6, scale: 1.026, duration: 15, ease: "sine.inOut" }, 0);
    },
    ambientLoop(tl) {
      tl.fromTo(".scan-band", { x: -180 }, { x: 1560, duration: 15, ease: "none" }, 0);
      tl.fromTo(".ambient-grid", { y: -80 }, { y: 80, duration: 15, ease: "none" }, 0);
      tl.to(".grain", { opacity: .24, duration: 7.5, yoyo: true, repeat: 1, ease: "sine.inOut" }, 0);
    },
    microActing(tl, selector, start, duration, amount) {
      tl.to(selector, { y: -amount, rotate: -0.9, duration: duration / 2, ease: "sine.inOut" }, start);
      tl.to(selector, { y: amount * .35, rotate: .5, duration: duration / 2, ease: "sine.inOut" }, start + duration / 2);
    },
    poseTransition(tl, outgoingSelector, incomingSelector, start, options) {
      const anticipation = options.anticipation || .12;
      const hold = options.hold || .7;
      const settle = options.settle || .18;
      const lean = options.lean || 10;
      tl.to(outgoingSelector, { x: -lean, rotate: -0.6, duration: anticipation, ease: "power2.inOut" }, start - anticipation);
      tl.to(outgoingSelector, { autoAlpha: 0, duration: .12, ease: "power1.out" }, start);
      tl.fromTo(incomingSelector, { autoAlpha: 0, x: lean, rotate: .8 }, { autoAlpha: 1, x: 0, rotate: 0, duration: settle, ease: "power2.out" }, start);
      tl.to(incomingSelector, { y: -4, duration: hold / 2, yoyo: true, repeat: 1, ease: "sine.inOut" }, start + settle);
    },
    feeStackSimmer(tl, start, duration) {
      tl.to(".fee-tag", { y: -6, x: 4, duration: .55, stagger: .09, yoyo: true, repeat: cycleRepeat(duration, 1.1), ease: "sine.inOut" }, start);
    },
    machineIdle(tl, start, duration) {
      tl.to(".machine-body", { x: 5, y: -3, duration: .42, yoyo: true, repeat: cycleRepeat(duration, .84), ease: "sine.inOut" }, start);
      tl.to(".pulse-node", { scale: 1.7, autoAlpha: .25, duration: .32, stagger: .11, yoyo: true, repeat: cycleRepeat(duration, .86), ease: "sine.inOut" }, start + .08);
      tl.to(".flow", { x: 18, duration: .7, yoyo: true, repeat: cycleRepeat(duration, 1.4), ease: "sine.inOut" }, start);
    },
    machineCausality(tl, start, options) {
      const duration = options.duration || 3.2;
      const pulses = options.pulses || 2;
      const feePush = options.feePush || -18;
      const feeScale = options.feeScale || 1.03;
      const accentColor = options.accentColor || "#d89a3a";
      const settleColor = options.settleColor || "#c85f3f";
      tl.to(".lever", { rotate: -34, duration: .22, yoyo: true, repeat: pulses, ease: "power2.inOut" }, start + .15);
      tl.to(".gear-a", { rotate: 360, transformOrigin: "50% 50%", duration, ease: "none", repeat: Math.max(0, pulses - 1) }, start);
      tl.to(".gear-b", { rotate: -360, transformOrigin: "50% 50%", duration: duration + .4, ease: "none", repeat: Math.max(0, pulses - 1) }, start);
      tl.to(".pulse-node", { scale: 2.05, autoAlpha: .38, duration: .22, stagger: .08, yoyo: true, repeat: 1, ease: "power2.inOut" }, start + .42);
      tl.to(".flow", { x: 30, borderColor: accentColor, duration: .26, yoyo: true, repeat: 1, ease: "power2.inOut" }, start + .62);
      tl.to(".fee-tag", { x: feePush, scale: feeScale, borderColor: settleColor, duration: .2, stagger: .07, yoyo: true, repeat: 1, ease: "power2.inOut" }, start + duration - .32);
    },
  };
}());
"""
    )


def write_scene_choreography(out_dir: Path) -> None:
    choreography = {
        "title": "Signal Room Fee Machine V2 Scene Choreography",
        "status": "review-only",
        "public_release": False,
        "composition_id": "fee-machine-v2-review",
        "rule": "Each beat must declare acting objective, primary motion, and overlap into the next beat.",
        "beats": list(SCENE_CHOREOGRAPHY),
    }
    (out_dir / "scene_choreography.json").write_text(json.dumps(choreography, indent=2) + "\n")


def write_index(out_dir: Path) -> None:
    (out_dir / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Signal Room Fee Machine V2 Review Scaffold</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #101820; font-family: Inter, Arial, sans-serif; }
    #fee-machine-v2-review {
      width: 1080px;
      height: 1920px;
      overflow: hidden;
      position: relative;
      background:
        radial-gradient(circle at 26% 18%, rgba(216,154,58,.22), transparent 28%),
        linear-gradient(180deg, #101820 0%, #131d26 100%);
      color: #f3eadb;
    }
    .grain { position: absolute; inset: 0; opacity: .16; background-image: repeating-linear-gradient(0deg, rgba(255,255,255,.03) 0 1px, transparent 1px 4px); }
    .ambient-grid { position: absolute; inset: -120px 0; opacity: .12; background-image: linear-gradient(rgba(216,154,58,.16) 1px, transparent 1px), linear-gradient(90deg, rgba(216,154,58,.1) 1px, transparent 1px); background-size: 90px 90px; }
    .scan-band { position: absolute; top: -160px; left: -380px; width: 220px; height: 2300px; opacity: .2; background: linear-gradient(90deg, transparent, rgba(216,154,58,.42), transparent); transform: rotate(12deg); }
    .stage { position: absolute; inset: 0; transform-origin: 50% 54%; }
    .safe { position: absolute; left: 90px; top: 180px; width: 900px; height: 1560px; border: 2px solid rgba(216,154,58,.18); }
    .character { position: absolute; left: -58px; bottom: 52px; width: 575px; height: 1022px; object-fit: contain; object-position: bottom left; filter: drop-shadow(0 22px 28px rgba(0,0,0,.36)); }
    .bill-card {
      position: absolute; left: 118px; top: 236px; width: 440px; padding: 32px; border-radius: 24px;
      background: #f3eadb; color: #16191d; box-shadow: 0 28px 70px rgba(0,0,0,.32);
      transform: rotate(-3deg);
    }
    .bill-eyebrow { font-size: 28px; font-weight: 800; color: #6f513d; text-transform: uppercase; }
    .total { margin-top: 12px; font-size: 92px; font-weight: 900; line-height: .9; }
    .rule { height: 10px; border-radius: 99px; background: #c85f3f; margin-top: 28px; }
    .fee-tag {
      position: absolute; left: 614px; width: 330px; padding: 22px 26px; border-radius: 18px;
      background: #17232d; border: 3px solid rgba(216,154,58,.82); color: #f3eadb;
      font-size: 34px; font-weight: 850; box-shadow: 0 18px 42px rgba(0,0,0,.28);
    }
    .fee-tag span { display: block; color: #d89a3a; font-size: 24px; margin-top: 7px; font-weight: 700; }
    .fee-base { top: 342px; }
    .fee-service { top: 458px; }
    .fee-processing { top: 574px; }
    .fee-platform { top: 690px; }
    .wall {
      position: absolute; right: 0; bottom: 250px; width: 620px; height: 790px;
      background: linear-gradient(90deg, #182631, #22313c); border-left: 5px solid rgba(216,154,58,.52);
      box-shadow: inset 24px 0 42px rgba(0,0,0,.28);
    }
    .machine { position: absolute; right: 72px; bottom: 338px; width: 490px; height: 598px; }
    .machine-body { position: absolute; inset: 120px 40px 40px 40px; border-radius: 34px; background: #1c2c38; border: 4px solid #d89a3a; }
    .gear { position: absolute; border: 16px solid #8fa0aa; border-radius: 50%; box-shadow: inset 0 0 0 16px #24313f; }
    .gear-a { left: 82px; top: 174px; width: 132px; height: 132px; }
    .gear-b { right: 84px; top: 238px; width: 166px; height: 166px; }
    .belt { position: absolute; left: 165px; top: 302px; width: 206px; height: 28px; border-radius: 99px; background: #c85f3f; transform: rotate(17deg); }
    .pulse-node { position: absolute; width: 28px; height: 28px; border-radius: 50%; background: #d89a3a; box-shadow: 0 0 28px rgba(216,154,58,.8); }
    .node-a { left: 142px; top: 292px; }
    .node-b { left: 252px; top: 325px; }
    .node-c { left: 360px; top: 364px; }
    .lever { position: absolute; right: 66px; top: 92px; width: 26px; height: 190px; background: #d89a3a; border-radius: 99px; transform-origin: bottom center; }
    .lever::before { content: ""; position: absolute; top: -44px; left: -31px; width: 88px; height: 88px; border-radius: 50%; background: #c85f3f; }
    .flow { position: absolute; left: 108px; bottom: 92px; font-size: 30px; color: #d89a3a; font-weight: 900; letter-spacing: 0; }
    .caption {
      position: absolute; left: 92px; right: 92px; bottom: 92px; padding: 28px 34px; border-radius: 22px;
      background: rgba(16,24,32,.86); border: 2px solid rgba(216,154,58,.45);
      font-size: 42px; line-height: 1.12; font-weight: 850;
    }
    .meta { position: absolute; right: 92px; top: 96px; color: #8fa0aa; font-size: 22px; font-weight: 750; text-transform: uppercase; }
  </style>
</head>
<body>
  <div id="fee-machine-v2-review" data-composition-id="fee-machine-v2-review" data-start="0" data-width="1080" data-height="1920" data-duration="15">
    <div class="grain"></div>
    <div class="ambient-grid"></div>
    <div class="scan-band"></div>
    <div class="safe"></div>
    <div class="meta">review scaffold / not public</div>
    <div class="stage">
      <img class="character pose-neutral" src="assets/poses/neutral_read.svg" alt="" />
      <img class="character pose-shock" src="assets/poses/bill_shock.svg" alt="" />
      <img class="character pose-look" src="assets/poses/look_to_machine.svg" alt="" />
      <img class="character pose-point" src="assets/poses/skeptical_point.svg" alt="" />
      <img class="character pose-lean" src="assets/poses/slight_lean.svg" alt="" />
      <section class="bill-card">
        <div class="bill-eyebrow">monthly bill</div>
        <div class="total">$83.41</div>
        <div class="rule"></div>
        <div class="rule" style="width: 68%; opacity: .55"></div>
      </section>
      <div class="fee-tag fee-base">base price <span>$59.00</span></div>
      <div class="fee-tag fee-service">service fee <span>$8.99</span></div>
      <div class="fee-tag fee-processing">processing fee <span>$4.43</span></div>
      <div class="fee-tag fee-platform">platform fee <span>$10.99</span></div>
      <div class="wall"></div>
      <div class="machine">
        <div class="machine-body"></div>
        <div class="gear gear-a"></div>
        <div class="gear gear-b"></div>
        <div class="belt"></div>
        <div class="pulse-node node-a"></div>
        <div class="pulse-node node-b"></div>
        <div class="pulse-node node-c"></div>
        <div class="lever"></div>
        <div class="flow">money -> processor -> platform</div>
      </div>
      <div class="caption">The final number looks simple because the moving parts stay behind the wall.</div>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <script src="motion_primitives.js"></script>
  <script>
    window.__timelines = window.__timelines || {};
    const tl = gsap.timeline({ paused: true });
    gsap.set([".pose-shock", ".pose-look", ".pose-point", ".pose-lean", ".fee-tag", ".wall", ".machine", ".caption"], { autoAlpha: 0 });
    gsap.set(".bill-card", { transformOrigin: "50% 50%" });
    gsap.set([".pose-neutral", ".pose-shock", ".pose-look", ".pose-point", ".pose-lean"], { transformOrigin: "46% 82%" });
    gsap.set(".stage", { transformOrigin: "50% 54%" });

    // motion quality primitives: keep every beat alive while preserving review readability.
    SignalRoomMotion.continuousDrift(tl);
    SignalRoomMotion.ambientLoop(tl);
    SignalRoomMotion.microActing(tl, ".pose-neutral", .2, 2.1, 10);
    SignalRoomMotion.microActing(tl, ".pose-shock", 2.45, 2.3, 14);
    SignalRoomMotion.microActing(tl, ".pose-look", 5.15, 3.1, 11);
    SignalRoomMotion.microActing(tl, ".pose-point", 8.65, 3.0, 13);
    SignalRoomMotion.microActing(tl, ".pose-lean", 12.1, 2.6, 9);
    SignalRoomMotion.poseTransition(tl, ".pose-neutral", ".pose-shock", 2.35, { anticipation: .12, hold: 2.4, settle: .18, lean: 12 });
    SignalRoomMotion.poseTransition(tl, ".pose-shock", ".pose-look", 5.0, { anticipation: .14, hold: 3.1, settle: .2, lean: 10 });
    SignalRoomMotion.poseTransition(tl, ".pose-look", ".pose-point", 8.5, { anticipation: .14, hold: 3.0, settle: .2, lean: 14 });
    SignalRoomMotion.poseTransition(tl, ".pose-point", ".pose-lean", 12.0, { anticipation: .12, hold: 2.4, settle: .2, lean: 8 });
    SignalRoomMotion.feeStackSimmer(tl, 3.0, 11.5);
    SignalRoomMotion.machineIdle(tl, 5.8, 8.9);
    SignalRoomMotion.machineCausality(tl, 6.05, { duration: 3.2, pulses: 3, feePush: -18 });
    SignalRoomMotion.machineCausality(tl, 12.2, { duration: 1.1, pulses: 1, feePush: -8, feeScale: 1.04, settleColor: "#c85f3f" });

    tl.from(".bill-card", { y: -70, rotate: -8, autoAlpha: 0, duration: .55, ease: "power3.out" }, 0);
    tl.to(".total", { scale: 1.08, color: "#c85f3f", duration: .22, yoyo: true, repeat: 1, ease: "power2.inOut" }, 2.45);
    tl.fromTo(".fee-base", { x: -120, autoAlpha: 0 }, { x: 0, autoAlpha: 1, duration: .34, ease: "power3.out" }, 2.75);
    tl.fromTo(".fee-service", { x: -120, autoAlpha: 0 }, { x: 0, autoAlpha: 1, duration: .34, ease: "power3.out" }, 3.25);
    tl.fromTo(".fee-processing", { x: -120, autoAlpha: 0 }, { x: 0, autoAlpha: 1, duration: .34, ease: "power3.out" }, 3.75);
    tl.fromTo(".fee-platform", { x: -120, autoAlpha: 0 }, { x: 0, autoAlpha: 1, duration: .34, ease: "power3.out" }, 4.25);
    tl.fromTo(".wall", { x: 620, autoAlpha: 1 }, { x: 0, duration: .75, ease: "power3.inOut" }, 5.05);
    tl.fromTo(".machine", { x: 80, scale: .94, autoAlpha: 0 }, { x: 0, scale: 1, autoAlpha: 1, duration: .55, ease: "power2.out" }, 5.65);
    tl.fromTo(".caption", { y: 60, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: .42, ease: "power3.out" }, 10.2);
    window.__timelines["fee-machine-v2-review"] = tl;
  </script>
  <!-- Rhythm: hold -> split -> reveal -> mechanical pulse -> acting beat -> settle -->
</body>
</html>
"""
    )


def write_package_files(out_dir: Path, rig_dir: Path) -> dict[str, Any]:
    manifest = {
        "title": "Signal Room Fee Machine V2 HyperFrames Scaffold",
        "status": "review-only",
        "duration_seconds": 15,
        "format": "1080x1920",
        "source_rig": str(rig_dir),
        "rhythm": "hold -> split -> reveal -> mechanical pulse -> acting beat -> settle",
        "public_release": False,
        "required_review": [
            "run scripts/signal_room_video_env_gate.py before local render attempts",
            "run scripts/signal_room_scaffold_gate.py before HyperFrames preview",
            "run npx hyperframes lint",
            "run npx hyperframes inspect",
            "run scripts/signal_room_contact_sheet.py on the approved character candidate",
            "source or design SFX from audio_cue_sheet.json",
            "review sampled frames from retention_frame_plan.json",
            "run scripts/signal_room_retention_frame_gate.py on sampled proof frames",
            "review first frame and contact sheet",
            "replace v0 vector rig if Blender/Moho candidate passes",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "signal-room-fee-machine-v2-scaffold",
                "private": True,
                "scripts": {
                    "check": "npx --yes hyperframes@0.6.38 lint && npx --yes hyperframes@0.6.38 inspect",
                    "preview": "npx --yes hyperframes@0.6.38 preview",
                    "render": "npx --yes hyperframes@0.6.38 render --quality draft",
                },
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "README_REVIEW.md").write_text(
        """# Signal Room Fee Machine V2 HyperFrames Scaffold

Status: review-only. Do not publish.

This project assembles the 10-15 second fee-machine proof choreography using
the local `signal_room_adult_investigator_v0_20260528` vector rig as a temporary
character layer. It is a choreography scaffold, not final brand art.

Environment:

```bash
python ../../scripts/signal_room_video_env_gate.py
```

Temporary vector rig, if Blender/Moho frames are not ready:

```bash
python ../../scripts/signal_room_temp_rig_seed.py \
  --out ../signal_room_adult_investigator_v0_20260528
```

Run:

```bash
python ../../scripts/signal_room_scaffold_gate.py .
npm run check
npm run preview
```

Audio:

- `audio_cue_sheet.json` lists the production SFX bed and exact cue timings.
- Keep the proof understandable muted, but do not review it as production-value
  without the bill snap, fee ticks, wall thump, machine loop, and final lever
  click.
- If final sound design assets are blocked, `signal_room_audio_asset_seed.py`
  can create review-only placeholder WAV files for gate plumbing.
- Run `signal_room_audio_asset_gate.py` before treating the audio package as
  ready for editorial review.

Retention review:

- `retention_frame_plan.json` lists the five required sample frames for the
  proof contact sheet: ordinary bill, number split, machine reveal, acting read,
  and memory anchor.
- If those frames read like five similar cards, the proof fails.
- If HyperFrames frame export is blocked, `signal_room_proof_frame_seed.py`
  can create a review-only placeholder `proof_frames/` package for gate plumbing.
- After exporting those PNG frames, run `signal_room_retention_frame_gate.py`
  before reviewing the proof as retention-ready.

Replace `assets/poses/*.svg` with the approved Blender/Moho character frames
after the rig acting scorecard passes.

Build the rig acting contact sheet before replacement:

```bash
python ../../scripts/signal_room_contact_sheet.py \
  /path/to/review-package/character_frames/Suit_Male \
  --out /path/to/review-package/Suit_Male_contact_sheet.svg
```
"""
    )
    return manifest


def create_scaffold(rig_dir: Path, out_dir: Path) -> dict[str, Any]:
    pose_paths = require_poses(rig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_out = out_dir / "assets" / "poses"
    poses_out.mkdir(parents=True, exist_ok=True)
    for path in pose_paths:
        shutil.copy2(path, poses_out / path.name)
    manifest_path = rig_dir / "rig_manifest.json"
    if manifest_path.exists():
        shutil.copy2(manifest_path, out_dir / "source_rig_manifest.json")

    write_design(out_dir)
    write_expanded_prompt(out_dir)
    write_audio_cue_sheet(out_dir)
    write_retention_frame_plan(out_dir)
    write_motion_primitives(out_dir)
    write_motion_library(out_dir)
    write_scene_choreography(out_dir)
    write_index(out_dir)
    return write_package_files(out_dir, rig_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = create_scaffold(args.rig_dir, args.out)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
