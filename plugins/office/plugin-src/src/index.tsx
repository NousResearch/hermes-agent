/**
 * Hermes Office — 3D Dashboard Plugin
 *
 * A live 3D virtual office that visualizes all Hermes profiles/agents.
 * Each profile appears as a Claw3D character in the office — walking when
 * the gateway is running, standing when idle, typing at a desk when
 * processing a session.
 *
 * 3D character models and office environment ported from Claw3D
 * (iamlukethedev/Claw3D) — MIT License.
 * Copyright (c) 2026 Luke The Dev
 * Copyright (c) 2026 zhuohoudeputao (Hermes integration)
 *
 * ── Architecture ──────────────────────────────────────────────────
 * The dashboard host renders OfficeView with ITS OWN React (exposed via
 * window.__HERMES_PLUGIN_SDK__.React). OfficeView is a thin wrapper that
 * only manages data fetching + a container <div>. It uses the SDK's React
 * for its hooks and createElement so the dashboard's React sees a single
 * consistent React instance in its tree.
 *
 * The entire 3D scene (R3F <Canvas> + all children) is rendered into that
 * container div by a SEPARATE React root created with the BUNDLED
 * react-dom/client's createRoot. This avoids the React instance mismatch
 * that occurs when R3F's bundled react-reconciler tries to manage children
 * that were created by the dashboard's React — the mismatch caused the
 * reconciler's fiber tree to not match, resulting in a blank/transparent
 * canvas.
 *
 * Everything (React, react-dom, Three.js, R3F, drei) is bundled inline by
 * esbuild — no shims. The 3D scene's JSX uses the bundled React's automatic
 * JSX runtime, and R3F's reconciler uses the same bundled React instance.
 *
 * Built with esbuild into a single IIFE bundle.
 */

// ── Bundled imports (for the 3D scene rendered by the bundled React) ──
//
// These are used by OfficeScene and all child components. They all resolve
// to the SAME bundled React instance, so R3F's react-reconciler, the JSX
// runtime, and every hook call in the 3D tree share one React.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import type { Root } from "react-dom/client";

// React import for JSX used in the 3D scene (bundled, NOT the SDK's React).
// The automatic JSX runtime (jsxImportSource: "react") pulls from here.
import React, {
  useState as bundledUseState,
  useEffect as bundledUseEffect,
  useRef as bundledUseRef,
  useMemo as bundledUseMemo,
  Component as bundledComponent,
} from "react";

// Claw3D ported components — these use the bundled React via their own
// `import ... from "react"` statements, all resolved to the same bundle.
import { AgentModel } from "./claw3d/agents";
import { FloorAndWalls, WallPictures } from "./claw3d/environment";
import {
  buildNavGrid,
  advanceAgent,
  astar,
  setCustomRoamPoints,
  type NavGrid,
} from "./claw3d/navigation";
import {
  DeskCubicleModel,
  ChairModel,
  WhiteboardModel,
  KanbanBoardModel,
  PlantModel,
  BookshelfModel,
  WaterCoolerModel,
  CoffeeMachineModel,
  LampModel,
  VendingMachineModel,
  TrashCanModel,
  RoundTableModel,
  TableRectModel,
  CouchModel,
  BeanbagModel,
  FridgeModel,
  StoveModel,
  CabinetModel,
  WallCabinetModel,
  MicrowaveModel,
  SinkModel,
  DishwasherModel,
  ComputerModel,
  KeyboardModel,
  MouseModel,
  ClockModel,
  PingPongModel,
  AtmModel,
  PhoneBoothModel,
  SmsBoothModel,
  ServerRackModel,
  ServerTerminalModel,
  WallModel,
  DoorModel,
  TreadmillModel,
  WeightBenchModel,
  DumbbellRackModel,
  RowingMachineModel,
  KettlebellRackModel,
  ExerciseBikeModel,
  PunchingBagModel,
  YogaMatModel,
  EaselModel,
  QaTerminalModel,
  DeviceRackModel,
  TestBenchModel,
  RugModel,
  TvModel,
  DualMonitorModel,
  CoffeeBarModel,
  SnackShelfModel,
  PosterModel,
  TrophyModel,
  RubiksCubeModel,
  PendantLampModel,
} from "./claw3d/furniture";
import { DayNightCycle } from "./claw3d/cameraLighting";
import {
  DEFAULT_FURNITURE,
  DESK_POSITIONS,
  ROAM_POINTS,
} from "./claw3d/furnitureDefaults";
import type { RenderAgent, FurnitureItem } from "./claw3d/types";
import { createDefaultAgentAvatarProfile } from "./claw3d/profile";
import { toWorld } from "./claw3d/geometry";

// ── Dashboard SDK React (for OfficeView, rendered by the host) ──────
//
// OfficeView is registered with window.__HERMES_PLUGINS__.register().
// The dashboard host renders it with the dashboard's React (from the SDK).
// Therefore OfficeView's hooks and createElement MUST come from the SDK's
// React, not the bundled one — otherwise the host's React sees an
// "Invalid hook call" because the component would be calling hooks from a
// different React instance than the one currently rendering it.
//
// We read the SDK once at module load; OfficeView closures over this.
const SDK: any =
  typeof window !== "undefined"
    ? (window as any).__HERMES_PLUGIN_SDK__
    : undefined;
const DReact: any = SDK ? SDK.React : React; // fallback to bundled if no SDK

// ── Error boundary (uses bundled React — rendered inside the 3D root) ──

class OfficeErrorBoundary extends bundledComponent<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: any) {
    console.error("[Office] 3D render error caught by boundary:", error.message, error.stack, info);
  }
  render() {
    if (this.state.hasError) {
      return this.props.fallback || React.createElement("div",
        { style: { padding: 16, color: "#ef4444", fontSize: 14 } },
        "[Office] 3D render error: " + (this.state.error?.message || "unknown"));
    }
    return this.props.children;
  }
}

// ── WebGL detection ─────────────────────────────────────────────

function hasWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl") || canvas.getContext("experimental-webgl"));
  } catch {
    return false;
  }
}

// ── Layout data types ─────────────────────────────────────────────

interface LayoutData {
  version: number;
  name: string;
  desk_positions: Array<{ x: number; y: number; facing: number; deskId: string }>;
  roam_points: Array<{ x: number; y: number }>;
  furniture: Array<Record<string, any>>;
}

// ── Snapshot data types ──────────────────────────────────────────

interface SnapshotData {
  timestamp: string;
  summary: {
    total_profiles: number;
    active_gateways: number;
    active_sessions: number;
    kanban_tasks_active: number;
    cron_jobs_enabled: number;
  };
  profiles: Array<{
    name: string;
    model: string;
    gateway_running: boolean;
    skill_count: number;
    description: string;
    is_default: boolean;
  }>;
  sessions: Array<{
    id: string;
    source: string;
    is_active: boolean;
    title: string;
    last_active: number;
  }>;
  kanban: Array<{
    id: string;
    title: string;
    status: string;
    assignee: string;
  }>;
  cron: Array<{
    id: string;
    name: string;
    schedule: string;
    enabled: boolean;
    last_status: string;
  }>;
}

// ── Map snapshot data to RenderAgent objects ─────────────────────
//
// Hermes profile → Claw3D RenderAgent mapping:
//   gateway_running + no active session → "walking" (agent walks around)
//   !gateway_running                    → "standing" (agent stands idle)
//   has active session                  → "sitting" (agent sits at desk, typing)

function profileToRenderAgent(
  profile: SnapshotData["profiles"][0],
  index: number,
  sessions: SnapshotData["sessions"],
  kanban: SnapshotData["kanban"],
  frameRef: React.MutableRefObject<number>,
  deskPositions: typeof DESK_POSITIONS,
  roamPoints: typeof ROAM_POINTS,
): RenderAgent {
  // Only the profile with a running gateway can have active sessions —
  // sessions come from that gateway's session DB. A stopped gateway
  // means no sessions, regardless of stale "is_active" flags in the DB.
  // We also filter out stale sessions (last_active > 1 hour ago) that
  // were never properly closed — these are zombie CLI sessions from days
  // ago that still have is_active=true in the DB.
  const now = Date.now() / 1000;
  const hasActiveSession =
    profile.gateway_running &&
    sessions.some(
      (s) =>
        s.is_active &&
        (s.source === "telegram" || s.source === "cli") &&
        now - s.last_active < 3600, // only count sessions active in the last hour
    );
  const hasKanbanTask = kanban.some((t) => t.assignee === profile.name);

  // Determine state:
  //   sitting  — active session only (agent is actively working right now)
  //   walking  — no active session (agent roams the office, even if assigned a task)
  let state: RenderAgent["state"] = "walking";
  if (hasActiveSession) {
    state = "sitting";
  }

  // Determine status:
  //   working — actively in a session (talking to a user right now)
  //   idle    — not in a session (roaming, even if assigned a task)
  let status: RenderAgent["status"] = "idle";
  if (hasActiveSession) {
    status = "working";
  }

  // Position: sitting agents go to their desk, walking agents roam, standing agents at desk
  let x: number, y: number, facing: number;
  if (state === "sitting") {
    const desk = deskPositions[index % deskPositions.length];
    x = desk.x;
    y = desk.y;
    facing = desk.facing;
  } else if (state === "walking") {
    // Walking agents start at their roam point — AgentAnimationUpdater
    // updates their position every frame
    const roamIdx = index % roamPoints.length;
    const roam = roamPoints[roamIdx];
    x = roam.x;
    y = roam.y;
    facing = 0;
  } else {
    // Standing — at desk but not sitting
    const desk = deskPositions[index % deskPositions.length];
    x = desk.x;
    y = desk.y - 40;
    facing = desk.facing;
  }

  // Generate a stable color from profile name
  const colors = [
    "#4a90d9", "#e0533d", "#3dce05", "#d97e4a",
    "#9b4ad9", "#4ad9b0", "#d94a8e", "#4ad9d9",
  ];
  const color = colors[index % colors.length];

  // Create avatar profile from agent name for procedural appearance
  const avatarProfile = createDefaultAgentAvatarProfile(profile.name);

  return {
    id: `profile-${profile.name}`,
    name: profile.name,
    subtitle: profile.model,
    status,
    color,
    item: "desk",
    avatarProfile,
    x,
    y,
    targetX: x,
    targetY: y,
    path: [],
    facing,
    frame: frameRef.current,
    walkSpeed: 0.3,
    phaseOffset: index * 10,
    state,
  };
}

// ── Furniture renderer ───────────────────────────────────────────

function FurnitureRenderer({ items, kanbanTaskCount }: { items: FurnitureItem[]; kanbanTaskCount: number }) {
  return (
    <>
      {items.map((item) => {
        switch (item.type) {
          case "desk_cubicle":
            return <DeskCubicleModel key={item._uid} item={item} />;
          case "chair":
            return <ChairModel key={item._uid} item={item} />;
          case "whiteboard":
            return <WhiteboardModel key={item._uid} item={item} />;
          case "kanban_board":
            return <KanbanBoardModel key={item._uid} item={item} taskCount={kanbanTaskCount} />;
          case "plant":
            return <PlantModel key={item._uid} item={item} />;
          case "bookshelf":
            return <BookshelfModel key={item._uid} item={item} />;
          case "water_cooler":
            return <WaterCoolerModel key={item._uid} item={item} />;
          case "coffee_machine":
            return <CoffeeMachineModel key={item._uid} item={item} />;
          case "lamp":
            return <LampModel key={item._uid} item={item} />;
          case "vending":
            return <VendingMachineModel key={item._uid} item={item} />;
          case "trash":
            return <TrashCanModel key={item._uid} item={item} />;
          case "round_table":
            return <RoundTableModel key={item._uid} item={item} />;
          case "table_rect":
            return <TableRectModel key={item._uid} item={item} />;
          case "couch":
            return <CouchModel key={item._uid} item={item} />;
          case "beanbag":
            return <BeanbagModel key={item._uid} item={item} />;
          case "fridge":
            return <FridgeModel key={item._uid} item={item} />;
          case "stove":
            return <StoveModel key={item._uid} item={item} />;
          case "cabinet":
            return <CabinetModel key={item._uid} item={item} />;
          case "wall_cabinet":
            return <WallCabinetModel key={item._uid} item={item} />;
          case "microwave":
            return <MicrowaveModel key={item._uid} item={item} />;
          case "sink":
            return <SinkModel key={item._uid} item={item} />;
          case "dishwasher":
            return <DishwasherModel key={item._uid} item={item} />;
          case "computer":
            return <ComputerModel key={item._uid} item={item} />;
          case "keyboard":
            return <KeyboardModel key={item._uid} item={item} />;
          case "mouse":
            return <MouseModel key={item._uid} item={item} />;
          case "clock":
            return <ClockModel key={item._uid} item={item} />;
          case "pingpong":
            return <PingPongModel key={item._uid} item={item} />;
          case "atm":
            return <AtmModel key={item._uid} item={item} />;
          case "phone_booth":
            return <PhoneBoothModel key={item._uid} item={item} />;
          case "sms_booth":
            return <SmsBoothModel key={item._uid} item={item} />;
          case "server_rack":
            return <ServerRackModel key={item._uid} item={item} />;
          case "server_terminal":
            return <ServerTerminalModel key={item._uid} item={item} />;
          case "wall":
            return <WallModel key={item._uid} item={item} />;
          case "door":
            return <DoorModel key={item._uid} item={item} />;
          case "treadmill":
            return <TreadmillModel key={item._uid} item={item} />;
          case "weight_bench":
            return <WeightBenchModel key={item._uid} item={item} />;
          case "dumbbell_rack":
            return <DumbbellRackModel key={item._uid} item={item} />;
          case "rowing_machine":
            return <RowingMachineModel key={item._uid} item={item} />;
          case "kettlebell_rack":
            return <KettlebellRackModel key={item._uid} item={item} />;
          case "exercise_bike":
            return <ExerciseBikeModel key={item._uid} item={item} />;
          case "punching_bag":
            return <PunchingBagModel key={item._uid} item={item} />;
          case "yoga_mat":
            return <YogaMatModel key={item._uid} item={item} />;
          case "easel":
            return <EaselModel key={item._uid} item={item} />;
          case "qa_terminal":
            return <QaTerminalModel key={item._uid} item={item} />;
          case "device_rack":
            return <DeviceRackModel key={item._uid} item={item} />;
          case "test_bench":
            return <TestBenchModel key={item._uid} item={item} />;
          case "rug":
            return <RugModel key={item._uid} item={item} />;
          case "tv":
            return <TvModel key={item._uid} item={item} />;
          case "dual_monitor":
            return <DualMonitorModel key={item._uid} item={item} />;
          case "coffee_bar":
            return <CoffeeBarModel key={item._uid} item={item} />;
          case "snack_shelf":
            return <SnackShelfModel key={item._uid} item={item} />;
          case "poster":
            return <PosterModel key={item._uid} item={item} />;
          case "trophy":
            return <TrophyModel key={item._uid} item={item} />;
          case "rubiks_cube":
            return <RubiksCubeModel key={item._uid} item={item} />;
          case "pendant_lamp":
            return <PendantLampModel key={item._uid} item={item} />;
          default:
            return null;
        }
      })}
    </>
  );
}

// ── Agent animation updater ──────────────────────────────────────
//
// Uses Claw3D's real navigation system: A* pathfinding on a nav grid built
// from furniture. Agents walk around desks and walls, not through them.
// When a walking agent's path is exhausted, a new roam point is picked and
// a fresh A* path is planned — exactly as Claw3D's useAgentTick does it.

function AgentAnimationUpdater({
  agentsRef,
  frameRef,
  furniture,
}: {
  agentsRef: React.MutableRefObject<RenderAgent[]>;
  frameRef: React.MutableRefObject<number>;
  furniture: FurnitureItem[];
}) {
  // Build nav grid from furniture (rebuilt when furniture changes)
  const gridRef = bundledUseRef<NavGrid | null>(null);
  bundledUseEffect(() => {
    gridRef.current = buildNavGrid(furniture);
  }, [furniture]);
  const nowRef = bundledUseRef(performance.now());

  useFrame((_, delta) => {
    frameRef.current += 1;
    const agents = agentsRef.current;
    if (!agents) return;
    const grid = gridRef.current;
    if (!grid) return;

    // Clamp delta — R3F can deliver huge deltas after tab switches
    const dt = Math.min(delta, 0.05);
    nowRef.current = performance.now();

    for (let i = 0; i < agents.length; i++) {
      const agent = agents[i];

      if (agent.state === "walking") {
        advanceAgent(agent, dt, grid, nowRef.current);
      } else if (agent.state === "sitting") {
        // If far from desk, walk there first instead of teleporting
        const dx = agent.targetX - agent.x;
        const dy = agent.targetY - agent.y;
        const dist = Math.hypot(dx, dy);
        if (dist > 5) {
          // Still walking to desk — use A* to navigate around furniture
          if (!agent.path || agent.path.length === 0) {
            agent.path = astar(agent.x, agent.y, agent.targetX, agent.targetY, grid);
          }
          if (agent.path && agent.path.length > 0) {
            advanceAgent(agent, dt, grid, nowRef.current);
            // Keep state as sitting so when the agent arrives, it sits
            agent.state = "sitting";
          } else {
            // No path found — just lerp directly
            const speed = 200 * dt;
            if (dist > speed) {
              agent.x += (dx / dist) * speed;
              agent.y += (dy / dist) * speed;
              agent.facing = Math.atan2(dx, dy);
            } else {
              agent.x = agent.targetX;
              agent.y = agent.targetY;
            }
          }
        } else {
          // At desk — update frame for typing animation
          agent.x = agent.targetX;
          agent.y = agent.targetY;
          agent.frame = frameRef.current;
        }
      } else if (agent.state === "standing") {
        // Standing agents shift weight slightly
        const t = frameRef.current * 0.012 + i * 0.5;
        agent.x = agent.targetX + Math.sin(t * 0.5) * 3;
        agent.y = agent.targetY + Math.cos(t * 0.3) * 2;
        agent.frame = frameRef.current;
      }
    }
  });
  return null;
}

// ── Camera controller ────────────────────────────────────────────

function CameraController() {
  const { camera } = useThree();
  // Position camera high enough to see the full office (world is ~32x32 units)
  const targetPos = bundledUseMemo(() => new THREE.Vector3(0, 18, 22), []);
  const targetLook = bundledUseMemo(() => new THREE.Vector3(0, 0.5, 0), []);

  useFrame(() => {
    camera.position.lerp(targetPos, 0.05);
    camera.lookAt(targetLook);
  });

  return null;
}

// ── Main 3D Scene ────────────────────────────────────────────────
//
// This component and ALL its children are rendered by the BUNDLED React
// (via createRoot in OfficeView's useEffect). They use the bundled React's
// hooks and JSX runtime. R3F's reconciler uses the same bundled React.

function OfficeScene({ data, layout }: { data: SnapshotData; layout: LayoutData | null }) {
  const frameRef = bundledUseRef(0);

  // Use layout from API, or fall back to hardcoded defaults
  const deskPositions = layout?.desk_positions ?? DESK_POSITIONS;
  const roamPoints = layout?.roam_points ?? ROAM_POINTS;

  // Sync custom roam points to the navigation system so advanceAgent
  // picks targets from the user's layout, not the hardcoded defaults.
  bundledUseEffect(() => {
    setCustomRoamPoints(roamPoints as typeof ROAM_POINTS);
  }, [roamPoints]);

  const furniture = bundledUseMemo(() => {
    if (layout?.furniture) {
      return layout.furniture.map((seed, i) => ({ ...seed, _uid: `layout_${i}` }) as FurnitureItem);
    }
    return DEFAULT_FURNITURE;
  }, [layout]);

  // Update agent data in-place — never replace the array or objects.
  // This prevents AgentModel's useFrame from seeing a different object
  // reference (which causes position jumps). We only update status/state
  // and target; the animation loop owns x/y/facing/path.
  const agentsRef = bundledUseRef<RenderAgent[]>([]);
  const agentLookupRef = bundledUseRef<Map<string, RenderAgent>>(new Map());

  // Initialise agents on first data load
  if (agentsRef.current.length === 0 && data.profiles.length > 0) {
    agentsRef.current = data.profiles.map((p, i) =>
      profileToRenderAgent(p, i, data.sessions, data.kanban, frameRef, deskPositions, roamPoints)
    );
    const map = new Map<string, RenderAgent>();
    agentsRef.current.forEach((a) => map.set(a.id, a));
    agentLookupRef.current = map;
  }

  // On subsequent polls, update status/state/targets in-place
  bundledUseEffect(() => {
    const agents = agentsRef.current;
    const map = agentLookupRef.current;
    data.profiles.forEach((p, i) => {
      const id = `profile-${p.name}`;
      let agent = map.get(id);
      if (!agent) {
        // New profile — create agent
        agent = profileToRenderAgent(p, i, data.sessions, data.kanban, frameRef, deskPositions, roamPoints);
        agents.push(agent);
        map.set(id, agent);
      } else {
        // Update in-place — only status/state, preserve position
        const now = Date.now() / 1000;
        const hasActiveSession =
          p.gateway_running &&
          data.sessions.some(
            (s) =>
              s.is_active &&
              (s.source === "telegram" || s.source === "cli") &&
              now - s.last_active < 3600,
          );
        const hasKanbanTask = data.kanban.some((t) => t.assignee === p.name);

        const newState: RenderAgent["state"] = hasActiveSession ? "sitting" : "walking";
        const newStatus: RenderAgent["status"] = hasActiveSession ? "working" : "idle";

        // If state changed, set new target so animation loop walks there
        if (agent.state !== newState) {
          agent.state = newState;
          if (newState === "sitting") {
            const desk = deskPositions[i % deskPositions.length];
            agent.targetX = desk.x;
            agent.targetY = desk.y;
            agent.path = []; // will be replanned by animation loop
          } else {
            // Walking — pick a roam point
            agent.path = [];
            const roam = roamPoints[i % roamPoints.length];
            agent.targetX = roam.x;
            agent.targetY = roam.y;
          }
        }
        agent.status = newStatus;
      }
    });
    // Remove agents for deleted profiles
    const currentIds = new Set(data.profiles.map((p) => `profile-${p.name}`));
    const remaining = agents.filter((a) => currentIds.has(a.id));
    agentsRef.current = remaining;
    // Rebuild lookup map
    const newMap = new Map<string, RenderAgent>();
    remaining.forEach((a) => newMap.set(a.id, a));
    agentLookupRef.current = newMap;
  }, [data, deskPositions, roamPoints]);

  const kanbanTaskCount = data.kanban.length;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 10, 5]} intensity={1.0} />

      <DayNightCycle />
      <AgentAnimationUpdater agentsRef={agentsRef} frameRef={frameRef} furniture={furniture} />

      {/* Office environment */}
      <FloorAndWalls />
      <WallPictures />

      {/* Furniture */}
      <FurnitureRenderer items={furniture} kanbanTaskCount={kanbanTaskCount} />

      {/* Agent characters — using Claw3D's full AgentModel */}
      {/* Render from data.profiles (triggers re-render when profiles change).
          AgentModel reads live position from agentsRef every frame. */}
      {data.profiles.map((p, i) => {
        const id = `profile-${p.name}`;
        const agent = agentLookupRef.current.get(id);
        if (!agent) return null;
        return (
          <AgentModel
            key={id}
            agentId={id}
            name={p.name}
            subtitle={p.model}
            status={agent.status}
            color={agent.color}
            appearance={agent.avatarProfile}
            agentsRef={agentsRef}
            agentLookupRef={agentLookupRef}
          />
        );
      })}

      <OrbitControls
        target={[0, 0.5, 0]}
        minDistance={8}
        maxDistance={50}
        maxPolarAngle={Math.PI / 2.1}
        enablePan={true}
      />
    </>
  );
}

// ── Summary overlay (2D HTML over the 3D canvas) ─────────────────
//
// This is rendered inside the bundled React root (as a sibling of Canvas),
// so it uses the bundled React's JSX. It's pure 2D HTML, no R3F.

function SummaryOverlay({ data }: { data: SnapshotData }) {
  const s = data.summary;
  return (
    <div style={{
      position: "absolute",
      top: 12,
      left: 12,
      zIndex: 10,
      display: "flex",
      gap: 16,
      padding: "8px 16px",
      background: "rgba(15, 15, 30, 0.85)",
      borderRadius: 8,
      fontSize: 13,
      color: "#e5e7eb",
      fontFamily: "system-ui, sans-serif",
      pointerEvents: "none",
    }}>
      <span><strong>Office</strong></span>
      <span>{s.total_profiles} profiles</span>
      <span style={{ color: "#22c55e" }}>{s.active_gateways} gateways</span>
      <span>{s.active_sessions} active sessions</span>
      {s.kanban_tasks_active > 0 && (
        <span style={{ color: "#f59e0b" }}>{s.kanban_tasks_active} kanban</span>
      )}
    </div>
  );
}

// ── The 3D scene element tree (bundled React JSX) ─────────────────
//
// This function returns JSX that is rendered by the bundled React's
// createRoot. It wraps Canvas + SummaryOverlay in a relative-positioned
// container div. The error boundary catches R3F/three errors and shows
// the 2D fallback.

function Office3DRoot({ data, layout }: { data: SnapshotData; layout: LayoutData | null }) {
  return (
    <OfficeErrorBoundary fallback={<Fallback2DView data={data} />}>
      <div style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}>
        <SummaryOverlay data={data} />
        <Canvas
          shadows
          camera={{ position: [0, 18, 22], fov: 50, near: 0.1, far: 200 }}
          style={{ background: "#1a1a2e" }}
          frameloop="always"
        >
          <OfficeScene data={data} layout={layout} />
        </Canvas>
      </div>
    </OfficeErrorBoundary>
  );
}

// ── Main plugin component ────────────────────────────────────────
//
// OfficeView is rendered by the DASHBOARD'S React (the host calls
// React.createElement(OfficeView)). Therefore it MUST use the SDK's React
// for hooks and createElement — the dashboard's React is the one managing
// this component's fiber.
//
// OfficeView's only job is:
//   1. Fetch snapshot data (using SDK hooks + SDK.fetchJSON)
//   2. Render a thin container <div> with a ref (using SDK React)
//   3. In useEffect, create a BUNDLED React root and render the 3D scene
//      into the container div. The bundled React owns the entire 3D tree.

function OfficeView() {
  // ── Hooks from the DASHBOARD's React (SDK) ──
  const [data, setData] = DReact.useState(null);
  const [layout, setLayout] = DReact.useState(null);
  const [loading, setLoading] = DReact.useState(true);
  const [error, setError] = DReact.useState(null);
  const containerRef = DReact.useRef(null);
  const rootRef = DReact.useRef(null);
  const pollRef = DReact.useRef(null);

  const fetchSnapshot = DReact.useCallback(() => {
    const sdk = (window as any).__HERMES_PLUGIN_SDK__;
    if (!sdk) return;
    sdk.fetchJSON("/api/plugins/office/snapshot")
      .then((res: SnapshotData) => {
        setData(res);
        setError(null);
      })
      .catch((e: Error) => {
        setError(String(e));
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  // Fetch the office layout once on mount
  DReact.useEffect(() => {
    const sdk = (window as any).__HERMES_PLUGIN_SDK__;
    if (!sdk) return;
    sdk.fetchJSON("/api/plugins/office/layout")
      .then((res: LayoutData) => {
        setLayout(res);
      })
      .catch(() => {
        // Layout fetch failed — OfficeScene will use hardcoded defaults
      });
  }, []);

  DReact.useEffect(() => {
    fetchSnapshot();
    pollRef.current = setInterval(fetchSnapshot, 5000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [fetchSnapshot]);

  // ── Render the 3D scene with the BUNDLED React ──
  // When data arrives, create a separate React root (bundled react-dom/client)
  // and render the entire 3D tree into the container div. This isolates the
  // bundled React from the dashboard's React — no instance mismatch.
  DReact.useEffect(() => {
    if (!containerRef.current || !data) return;

    // Create the root ONCE — on subsequent data updates, just re-render
    // into the existing root. Unmounting/remounting every poll (5s) would
    // destroy the WebGL canvas and rebuild the entire scene, causing the
    // visible "refresh" flicker.
    if (!rootRef.current) {
      rootRef.current = createRoot(containerRef.current);
    }
    rootRef.current.render(React.createElement(Office3DRoot, { data, layout }));

    // No cleanup here — unmount is handled by the separate effect below
    // so the root survives data updates.
  }, [data, layout]);

  // Unmount the bundled React root only when the component truly unmounts
  DReact.useEffect(() => {
    return () => {
      if (rootRef.current) {
        rootRef.current.unmount();
        rootRef.current = null;
      }
    };
  }, []);

  // ── Loading / error / no-data states (dashboard React) ──
  if (error) {
    return DReact.createElement("div",
      { style: { padding: 16, color: "#ef4444", fontSize: 14 } },
      "Office plugin error: " + error);
  }

  if (!data && loading) {
    return DReact.createElement("div",
      { style: { padding: 16, color: "#9ca3af", fontSize: 14 } },
      "Loading office…");
  }

  if (!data) {
    return DReact.createElement("div",
      { style: { padding: 16, color: "#9ca3af", fontSize: 14 } },
      "No data available.");
  }

  // 2D fallback when WebGL is not available (headless browsers, etc.)
  if (!hasWebGL()) {
    return DReact.createElement(Fallback2DView, { data });
  }

  // ── Container div: the dashboard React renders ONLY this div. ──
  // The bundled React renders the 3D scene into it via createRoot (above).
  return DReact.createElement("div",
    {
      ref: containerRef,
      style: { width: "100%", height: "calc(100vh - 50px)" },
    });
}

// ── 2D Fallback view (when WebGL is unavailable) ─────────────────
//
// Fallback2DView can be rendered by either React instance — it's pure
// createElement with no hooks, so it works in both. When rendered by the
// dashboard's React (WebGL unavailable path above), it uses DReact; when
// used as the error boundary fallback inside the 3D root, it uses the
// bundled React's JSX. Both work because it only uses createElement (no
// hooks, no fiber-dependent features).

function Fallback2DView({ data }: { data: SnapshotData }) {
  const s = data.summary;
  const deskColors: Record<string, string> = {
    working: "#22c55e",
    streaming: "#3b82f6",
    idle: "#6b7280",
    error: "#ef4444",
  };
  const deskLabels: Record<string, string> = {
    working: "Working",
    streaming: "Streaming",
    idle: "Idle",
    error: "Error",
  };

  return React.createElement("div",
    { style: { padding: 24, fontFamily: "system-ui, sans-serif", color: "#e5e7eb" } },
    // Summary header
    React.createElement("div",
      { style: { display: "flex", gap: 16, marginBottom: 24, fontSize: 14 } },
      React.createElement("strong", null, "Office"),
      React.createElement("span", null, `${s.total_profiles} profiles`),
      React.createElement("span", { style: { color: "#22c55e" } }, `${s.active_gateways} gateways`),
      React.createElement("span", null, `${s.active_sessions} active sessions`),
      s.kanban_tasks_active > 0 && React.createElement("span", { style: { color: "#f59e0b" } }, `${s.kanban_tasks_active} kanban`),
    ),
    // Agent cards grid
    React.createElement("div",
      { style: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", gap: 16 } },
      ...data.profiles.map((p, i) => {
        const now = Date.now() / 1000;
        const hasActiveSession =
          p.gateway_running &&
          data.sessions.some(
            (sess) =>
              sess.is_active &&
              (sess.source === "telegram" || sess.source === "cli") &&
              now - sess.last_active < 3600,
          );
        const hasKanbanTask = data.kanban.some((t) => t.assignee === p.name);
        let status = "idle";
        if (hasActiveSession) status = "streaming";
        else if (hasKanbanTask) status = "working";
        const color = deskColors[status] || "#6b7280";
        const stateLabel = hasActiveSession ? "Sitting (typing)" : p.gateway_running ? "Walking" : "Standing";
        return React.createElement("div",
          {
            key: p.name,
            style: {
              background: "rgba(15, 15, 30, 0.85)",
              border: `2px solid ${color}`,
              borderRadius: 8,
              padding: 16,
            },
          },
          React.createElement("div", { style: { fontSize: 16, fontWeight: "bold", marginBottom: 8 } }, p.name),
          React.createElement("div", { style: { fontSize: 12, color: "#9ca3af", marginBottom: 4 } }, p.model),
          React.createElement("div",
            { style: { display: "flex", alignItems: "center", gap: 6, fontSize: 12 } },
            React.createElement("span", { style: { width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" } }),
            deskLabels[status] || status,
          ),
          React.createElement("div", { style: { fontSize: 11, color: "#6b7280", marginTop: 8 } },
            `State: ${stateLabel}`),
          React.createElement("div", { style: { fontSize: 11, color: "#6b7280" } },
            p.gateway_running ? "● Gateway running" : "○ Gateway stopped"),
          React.createElement("div", { style: { fontSize: 11, color: "#6b7280" } },
            `${p.skill_count} skills`),
        );
      }),
    ),
    // Kanban tasks
    data.kanban.length > 0 && React.createElement("div",
      { style: { marginTop: 24 } },
      React.createElement("h3", { style: { fontSize: 14, color: "#9ca3af", marginBottom: 8 } }, "Kanban Tasks"),
      ...data.kanban.map((t) => React.createElement("div",
        { key: t.id, style: { fontSize: 12, marginBottom: 4 } },
        `[${t.status}] ${t.title}`,
      )),
    ),
    // Cron jobs
    data.cron.length > 0 && React.createElement("div",
      { style: { marginTop: 24 } },
      React.createElement("h3", { style: { fontSize: 14, color: "#9ca3af", marginBottom: 8 } }, "Cron Jobs"),
      ...data.cron.map((c) => React.createElement("div",
        { key: c.id, style: { fontSize: 12, marginBottom: 4 } },
        `${c.name} — ${c.schedule} — ${c.last_status || "pending"}`,
      )),
    ),
  );
}

// ── Register with the dashboard plugin system ────────────────────
//
// OfficeView is registered with the dashboard's plugin registry. The
// dashboard host renders it with the dashboard's React. OfficeView itself
// uses the SDK's React for hooks, and delegates the 3D scene to the
// bundled React via createRoot.

if (typeof window !== "undefined" && (window as any).__HERMES_PLUGINS__) {
  (window as any).__HERMES_PLUGINS__.register("office", OfficeView);
}
