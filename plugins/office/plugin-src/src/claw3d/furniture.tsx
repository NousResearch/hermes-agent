/**
 * Claw3D office furniture — adapted from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 * Copyright (c) 2026 zhuohoudeputao (Hermes integration)
 *
 * Office furniture using primitive Three.js geometry (no GLB loading —
 * the plugin bundles inline and doesn't have access to Claw3D's asset files).
 * Includes: desks, chairs, computers, whiteboards, kanban boards, plants,
 * bookshelves, cabinets, coffee machine, fridge, water cooler, lamps.
 */

import { memo } from "react";
import { Billboard, Text } from "@react-three/drei";
import { SCALE } from "./constants";
import { toWorld } from "./geometry";
import { getItemBaseSize, getItemRotationRadians } from "./navigation";
import type { FurnitureItem } from "./types";

/** Desk with cubicle walls, monitor, keyboard, mouse */
export const DeskCubicleModel = memo(function DeskCubicleModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const w = (item.w ?? 100) * SCALE;
  const d = (item.h ?? 55) * SCALE;

  return (
    <group position={[wx, 0, wz]}>
      <group position={[0, 0, 0]}>
        {/* Desk surface */}
        <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.04, d]} />
          <meshStandardMaterial color="#8b5e32" roughness={0.6} metalness={0.08} />
        </mesh>
        {/* Desk legs */}
        <mesh position={[-w / 2 + 0.03, 0.3, -d / 2 + 0.03]}>
          <boxGeometry args={[0.04, 0.6, 0.04]} />
          <meshStandardMaterial color="#3c2410" roughness={0.8} />
        </mesh>
        <mesh position={[w / 2 - 0.03, 0.3, -d / 2 + 0.03]}>
          <boxGeometry args={[0.04, 0.6, 0.04]} />
          <meshStandardMaterial color="#3c2410" roughness={0.8} />
        </mesh>
        <mesh position={[-w / 2 + 0.03, 0.3, d / 2 - 0.03]}>
          <boxGeometry args={[0.04, 0.6, 0.04]} />
          <meshStandardMaterial color="#3c2410" roughness={0.8} />
        </mesh>
        <mesh position={[w / 2 - 0.03, 0.3, d / 2 - 0.03]}>
          <boxGeometry args={[0.04, 0.6, 0.04]} />
          <meshStandardMaterial color="#3c2410" roughness={0.8} />
        </mesh>
        {/* Monitor */}
        <mesh position={[0, 0.78, -d / 2 + 0.06]} castShadow>
          <boxGeometry args={[w * 0.5, 0.22, 0.02]} />
          <meshStandardMaterial
            color="#0f0f1e"
            emissive="#1a1a3e"
            emissiveIntensity={0.4}
          />
        </mesh>
        <mesh position={[0, 0.62, -d / 2 + 0.04]}>
          <boxGeometry args={[0.04, 0.08, 0.04]} />
          <meshStandardMaterial color="#1a1a1a" />
        </mesh>
        {/* Keyboard */}
        <mesh position={[0, 0.625, 0.02]}>
          <boxGeometry args={[0.18, 0.015, 0.07]} />
          <meshStandardMaterial color="#2e333d" roughness={0.85} />
        </mesh>
        {/* Mouse */}
        <mesh position={[0.14, 0.625, 0.03]} scale={[1, 0.38, 0.72]}>
          <sphereGeometry args={[0.03, 8, 6]} />
          <meshStandardMaterial color="#d0cecc" roughness={0.6} />
        </mesh>
      </group>
    </group>
  );
});

/** Office chair */
export const ChairModel = memo(function ChairModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = ((item.facing ?? 0) * Math.PI) / 180;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Seat */}
      <mesh position={[0, 0.32, 0]} castShadow>
        <boxGeometry args={[0.22, 0.04, 0.22]} />
        <meshStandardMaterial color="#4a5568" roughness={0.7} />
      </mesh>
      {/* Backrest */}
      <mesh position={[0, 0.5, -0.1]} castShadow>
        <boxGeometry args={[0.22, 0.36, 0.04]} />
        <meshStandardMaterial color="#4a5568" roughness={0.7} />
      </mesh>
      {/* Post */}
      <mesh position={[0, 0.16, 0]}>
        <cylinderGeometry args={[0.02, 0.02, 0.28, 8]} />
        <meshStandardMaterial color="#2d3748" metalness={0.4} roughness={0.4} />
      </mesh>
      {/* Base (5-star) */}
      {[0, 1, 2, 3, 4].map((i) => {
        const angle = (i / 5) * Math.PI * 2;
        return (
          <mesh
            key={i}
            position={[Math.cos(angle) * 0.08, 0.02, Math.sin(angle) * 0.08]}
            rotation={[0, angle, 0]}
          >
            <boxGeometry args={[0.16, 0.015, 0.03]} />
            <meshStandardMaterial color="#2d3748" metalness={0.4} roughness={0.4} />
          </mesh>
        );
      })}
      {/* Wheels */}
      {[0, 1, 2, 3, 4].map((i) => {
        const angle = (i / 5) * Math.PI * 2;
        return (
          <mesh
            key={`wheel-${i}`}
            position={[Math.cos(angle) * 0.15, 0.02, Math.sin(angle) * 0.15]}
          >
            <sphereGeometry args={[0.02, 8, 8]} />
            <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
          </mesh>
        );
      })}
    </group>
  );
});

/** Whiteboard on wall */
export const WhiteboardModel = memo(function WhiteboardModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = ((item.facing ?? 0) * Math.PI) / 180;
  const w = (item.w ?? 10) * SCALE;
  const h = (item.h ?? 60) * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <mesh position={[0, 0.72, 0]} castShadow receiveShadow>
        <boxGeometry args={[Math.max(w, 0.04), Math.max(h, 0.5), 0.03]} />
        <meshStandardMaterial color="#f4f2ee" roughness={0.3} />
      </mesh>
      {/* Frame */}
      <mesh position={[0, 0.72, -0.01]}>
        <boxGeometry args={[Math.max(w, 0.04) + 0.02, Math.max(h, 0.5) + 0.02, 0.02]} />
        <meshStandardMaterial color="#5a4030" roughness={0.7} />
      </mesh>
    </group>
  );
});

/** Kanban board with columns */
export const KanbanBoardModel = memo(function KanbanBoardModel({
  item,
  taskCount = 0,
}: {
  item: FurnitureItem;
  taskCount?: number;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = ((item.facing ?? 0) * Math.PI) / 180;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Board surface */}
      <mesh position={[0, 1.2, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.3, 0.85, 0.05]} />
        <meshStandardMaterial color="#1e293b" roughness={0.7} />
      </mesh>
      {/* Stand post */}
      <mesh position={[0, 0.5, 0]}>
        <boxGeometry args={[0.06, 1.0, 0.06]} />
        <meshStandardMaterial color="#334155" />
      </mesh>
      <mesh position={[0, 0.04, 0.1]}>
        <boxGeometry args={[0.5, 0.04, 0.2]} />
        <meshStandardMaterial color="#334155" />
      </mesh>
      {/* Column headers */}
      {[-0.42, 0, 0.42].map((x, i) => (
        <mesh key={`col-${i}`} position={[x, 1.5, 0.03]}>
          <planeGeometry args={[0.36, 0.12]} />
          <meshBasicMaterial
            color={i === 0 ? "#f59e0b" : i === 1 ? "#3b82f6" : "#22c55e"}
          />
        </mesh>
      ))}
      {/* Column areas */}
      {[-0.42, 0, 0.42].map((x, i) => (
        <mesh key={`col-bg-${i}`} position={[x, 1.15, 0.03]}>
          <planeGeometry args={[0.36, 0.6]} />
          <meshBasicMaterial
            color={i === 0 ? "#f59e0b" : i === 1 ? "#3b82f6" : "#22c55e"}
            transparent
            opacity={0.12}
          />
        </mesh>
      ))}
      <Billboard position={[0, 1.78, 0]}>
        <Text fontSize={0.08} color="#94a3b8" anchorX="center" anchorY="middle">
          {taskCount > 0 ? `Kanban (${taskCount})` : "Kanban"}
        </Text>
      </Billboard>
    </group>
  );
});

/** Potted plant */
export const PlantModel = memo(function PlantModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0, wz]}>
      {/* Pot */}
      <mesh position={[0, 0.1, 0]} castShadow>
        <cylinderGeometry args={[0.08, 0.06, 0.2, 12]} />
        <meshStandardMaterial color="#5d4037" roughness={0.84} />
      </mesh>
      <mesh position={[0, 0.21, 0]}>
        <cylinderGeometry args={[0.09, 0.08, 0.02, 12]} />
        <meshStandardMaterial color="#3e2723" roughness={0.8} />
      </mesh>
      {/* Foliage */}
      <mesh position={[0, 0.38, 0]} castShadow>
        <sphereGeometry args={[0.12, 12, 12]} />
        <meshStandardMaterial color="#3a7a3a" roughness={0.9} />
      </mesh>
      <mesh position={[0.05, 0.48, 0.03]} castShadow>
        <sphereGeometry args={[0.08, 10, 10]} />
        <meshStandardMaterial color="#4a8a4a" roughness={0.9} />
      </mesh>
      <mesh position={[-0.04, 0.45, -0.04]} castShadow>
        <sphereGeometry args={[0.07, 10, 10]} />
        <meshStandardMaterial color="#2a6a2a" roughness={0.9} />
      </mesh>
    </group>
  );
});

/** Bookshelf */
export const BookshelfModel = memo(function BookshelfModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = ((item.facing ?? 0) * Math.PI) / 180;
  const w = (item.w ?? 80) * SCALE;
  const h = (item.h ?? 120) * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[w / 2, 0, h / 2]}>
        {/* Frame */}
        <mesh position={[0, h / 2, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, h, 0.15]} />
          <meshStandardMaterial color="#5c3520" roughness={0.8} />
        </mesh>
        {/* Shelves with books */}
        {[0.15, 0.4, 0.65, 0.9].map((shelfY, shelfIdx) => (
          <group key={`shelf-${shelfIdx}`}>
            <mesh position={[0, shelfY * h, 0.04]}>
              <boxGeometry args={[w * 0.9, 0.015, 0.12]} />
              <meshStandardMaterial color="#4a2810" roughness={0.8} />
            </mesh>
            {/* Books */}
            {Array.from({ length: 8 }).map((_, i) => {
              const bookX = -w * 0.4 + (i + 0.5) * (w * 0.8 / 8);
              const colors = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#e74c3c", "#16a085", "#d4af37"];
              return (
                <mesh key={`book-${shelfIdx}-${i}`} position={[bookX, shelfY * h + 0.06, 0.04]}>
                  <boxGeometry args={[0.018, 0.1, 0.08]} />
                  <meshStandardMaterial color={colors[i % colors.length]} roughness={0.7} />
                </mesh>
              );
            })}
          </group>
        ))}
      </group>
    </group>
  );
});

/** Water cooler */
export const WaterCoolerModel = memo(function WaterCoolerModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0, wz]}>
      <mesh position={[0, 0.25, 0]} castShadow>
        <cylinderGeometry args={[0.1, 0.1, 0.5, 12]} />
        <meshStandardMaterial color="#3a5070" roughness={0.3} metalness={0.2} />
      </mesh>
      <mesh position={[0, 0.55, 0]}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial color="#4080c0" transparent opacity={0.7} roughness={0.1} />
      </mesh>
      <mesh position={[0, 0.05, 0]}>
        <boxGeometry args={[0.16, 0.04, 0.16]} />
        <meshStandardMaterial color="#2a3a50" roughness={0.5} />
      </mesh>
    </group>
  );
});

/** Coffee machine */
export const CoffeeMachineModel = memo(function CoffeeMachineModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const elevation = item.elevation ?? 0;

  return (
    <group position={[wx, elevation, wz]}>
      <mesh position={[0, 0.15, 0]} castShadow>
        <boxGeometry args={[0.12, 0.3, 0.14]} />
        <meshStandardMaterial color="#2d2d38" roughness={0.4} metalness={0.3} />
      </mesh>
      <mesh position={[0, 0.32, 0]}>
        <boxGeometry args={[0.1, 0.04, 0.12]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.3} metalness={0.4} />
      </mesh>
      <mesh position={[0, 0.08, 0.04]}>
        <boxGeometry args={[0.06, 0.02, 0.04]} />
        <meshStandardMaterial color="#8a8a8a" metalness={0.6} roughness={0.2} />
      </mesh>
    </group>
  );
});

/** Floor lamp */
export const LampModel = memo(function LampModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0, wz]}>
      <mesh position={[0, 0.02, 0]}>
        <cylinderGeometry args={[0.1, 0.12, 0.04, 12]} />
        <meshStandardMaterial color="#3a3a3a" roughness={0.5} metalness={0.4} />
      </mesh>
      <mesh position={[0, 0.7, 0]}>
        <cylinderGeometry args={[0.015, 0.02, 1.4, 8]} />
        <meshStandardMaterial color="#c8a060" roughness={0.3} metalness={0.6} />
      </mesh>
      <mesh position={[0, 1.42, 0]}>
        <coneGeometry args={[0.12, 0.16, 12]} />
        <meshStandardMaterial
          color="#f5e6b8"
          emissive="#f5e6b8"
          emissiveIntensity={0.5}
        />
      </mesh>
      <pointLight position={[0, 1.35, 0]} intensity={0.3} distance={2} color="#f5e6b8" />
    </group>
  );
});

/** Vending machine */
export const VendingMachineModel = memo(function VendingMachineModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = ((item.facing ?? 0) * Math.PI) / 180;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <mesh position={[0, 0.65, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.4, 1.3, 0.35]} />
        <meshStandardMaterial color="#404858" roughness={0.5} metalness={0.2} />
      </mesh>
      {/* Glass front */}
      <mesh position={[0, 0.7, 0.18]}>
        <boxGeometry args={[0.32, 0.9, 0.01]} />
        <meshStandardMaterial color="#1a2a3a" transparent opacity={0.4} roughness={0.1} />
      </mesh>
      {/* Shelves with items */}
      {[0.4, 0.65, 0.9].map((shelfY, shelfIdx) =>
        [-0.1, 0, 0.1].map((x, i) => (
          <mesh key={`vend-${shelfIdx}-${i}`} position={[x, shelfY, 0.15]}>
            <boxGeometry args={[0.05, 0.06, 0.05]} />
            <meshStandardMaterial
              color={["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6"][i % 5]}
            />
          </mesh>
        )),
      )}
    </group>
  );
});

/** Trash can */
export const TrashCanModel = memo(function TrashCanModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0, wz]}>
      <mesh position={[0, 0.115, 0]}>
        <cylinderGeometry args={[0.055, 0.042, 0.23, 10]} />
        <meshStandardMaterial color="#4a4e58" roughness={0.8} metalness={0.12} />
      </mesh>
      <mesh position={[0, 0.234, 0]}>
        <cylinderGeometry args={[0.057, 0.057, 0.01, 10]} />
        <meshStandardMaterial color="#363940" roughness={0.7} metalness={0.18} />
      </mesh>
    </group>
  );
});

// ════════════════════════════════════════════════════════════════
//  NEW FURNITURE MODELS — procedural Three.js primitives
//  (no GLB loading — everything is boxes, cylinders, spheres)
// ════════════════════════════════════════════════════════════════

/** Round table — circular top with center leg */
export const RoundTableModel = memo(function RoundTableModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width } = getItemBaseSize(item);
  const radius = (width / 2) * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[radius, 0, radius]}>
        {/* Table top */}
        <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
          <cylinderGeometry args={[radius, radius, 0.04, 24]} />
          <meshStandardMaterial color="#8b5e32" roughness={0.6} />
        </mesh>
        {/* Center post */}
        <mesh position={[0, 0.3, 0]} castShadow>
          <cylinderGeometry args={[0.03, 0.04, 0.6, 12]} />
          <meshStandardMaterial color="#5c3520" roughness={0.7} />
        </mesh>
        {/* Base */}
        <mesh position={[0, 0.02, 0]} castShadow>
          <cylinderGeometry args={[0.12, 0.14, 0.04, 16]} />
          <meshStandardMaterial color="#5c3520" roughness={0.7} />
        </mesh>
      </group>
    </group>
  );
});

/** Rectangular table */
export const TableRectModel = memo(function TableRectModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.04, d]} />
          <meshStandardMaterial color="#8b5e32" roughness={0.6} />
        </mesh>
        {/* 4 legs */}
        {[
          [-w / 2 + 0.03, -d / 2 + 0.03],
          [w / 2 - 0.03, -d / 2 + 0.03],
          [-w / 2 + 0.03, d / 2 - 0.03],
          [w / 2 - 0.03, d / 2 - 0.03],
        ].map(([lx, lz], i) => (
          <mesh key={`leg-${i}`} position={[lx, 0.3, lz]} castShadow>
            <boxGeometry args={[0.04, 0.6, 0.04]} />
            <meshStandardMaterial color="#3c2410" roughness={0.8} />
          </mesh>
        ))}
      </group>
    </group>
  );
});

/** Couch — sofa with base, backrest, arms */
export const CouchModel = memo(function CouchModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;
  const color = item.color ?? "#4a5568";

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Base / seat cushion */}
        <mesh position={[0, 0.25, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.15, d]} />
          <meshStandardMaterial color={color} roughness={0.8} />
        </mesh>
        {/* Backrest */}
        <mesh position={[0, 0.45, -d / 2 + 0.04]} castShadow>
          <boxGeometry args={[w, 0.35, 0.08]} />
          <meshStandardMaterial color={color} roughness={0.8} />
        </mesh>
        {/* Arms */}
        <mesh position={[-w / 2 + 0.03, 0.32, 0]} castShadow>
          <boxGeometry args={[0.06, 0.22, d]} />
          <meshStandardMaterial color={color} roughness={0.8} />
        </mesh>
        <mesh position={[w / 2 - 0.03, 0.32, 0]} castShadow>
          <boxGeometry args={[0.06, 0.22, d]} />
          <meshStandardMaterial color={color} roughness={0.8} />
        </mesh>
        {/* Feet */}
        {[
          [-w / 2 + 0.04, -d / 2 + 0.04],
          [w / 2 - 0.04, -d / 2 + 0.04],
          [-w / 2 + 0.04, d / 2 - 0.04],
          [w / 2 - 0.04, d / 2 - 0.04],
        ].map(([lx, lz], i) => (
          <mesh key={`foot-${i}`} position={[lx, 0.06, lz]}>
            <cylinderGeometry args={[0.015, 0.015, 0.12, 8]} />
            <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
          </mesh>
        ))}
      </group>
    </group>
  );
});

/** Beanbag — squashed sphere seat */
export const BeanbagModel = memo(function BeanbagModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const color = item.color ?? "#e65100";

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <mesh position={[0, 0.12, 0]} castShadow receiveShadow scale={[1, 0.6, 1]}>
        <sphereGeometry args={[0.18, 16, 12]} />
        <meshStandardMaterial color={color} roughness={0.95} />
      </mesh>
      {/* Backrest part */}
      <mesh position={[0, 0.22, -0.08]} castShadow scale={[0.8, 0.7, 0.6]}>
        <sphereGeometry args={[0.15, 12, 10]} />
        <meshStandardMaterial color={color} roughness={0.95} />
      </mesh>
    </group>
  );
});

/** Fridge — tall box with door handle */
export const FridgeModel = memo(function FridgeModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Body */}
        <mesh position={[0, 0.7, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 1.4, d]} />
          <meshStandardMaterial color="#d0d0d0" roughness={0.3} metalness={0.5} />
        </mesh>
        {/* Door divider line */}
        <mesh position={[0, 1.1, d / 2 + 0.001]}>
          <boxGeometry args={[w, 0.01, 0.002]} />
          <meshStandardMaterial color="#999" roughness={0.3} />
        </mesh>
        {/* Handle */}
        <mesh position={[w / 2 - 0.03, 0.7, d / 2 + 0.02]} castShadow>
          <boxGeometry args={[0.015, 0.4, 0.02]} />
          <meshStandardMaterial color="#333" metalness={0.6} roughness={0.2} />
        </mesh>
      </group>
    </group>
  );
});

/** Stove — box with circular burners on top */
export const StoveModel = memo(function StoveModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Body */}
        <mesh position={[0, 0.45, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.9, d]} />
          <meshStandardMaterial color="#2a2a2e" roughness={0.4} metalness={0.3} />
        </mesh>
        {/* Top surface */}
        <mesh position={[0, 0.901, 0]} receiveShadow>
          <boxGeometry args={[w * 0.95, 0.01, d * 0.95]} />
          <meshStandardMaterial color="#1a1a1e" roughness={0.2} metalness={0.5} />
        </mesh>
        {/* Burners */}
        {[
          [-w * 0.2, -d * 0.2],
          [w * 0.2, -d * 0.2],
          [-w * 0.2, d * 0.2],
          [w * 0.2, d * 0.2],
        ].map(([bx, bz], i) => (
          <mesh key={`burner-${i}`} position={[bx, 0.91, bz]}>
            <cylinderGeometry args={[0.04, 0.04, 0.01, 16]} />
            <meshStandardMaterial color="#0a0a0a" roughness={0.5} />
          </mesh>
        ))}
        {/* Oven door handle */}
        <mesh position={[0, 0.3, d / 2 + 0.01]}>
          <boxGeometry args={[w * 0.6, 0.02, 0.02]} />
          <meshStandardMaterial color="#555" metalness={0.6} roughness={0.2} />
        </mesh>
      </group>
    </group>
  );
});

/** Cabinet — storage box */
export const CabinetModel = memo(function CabinetModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const elevation = item.elevation ?? 0;
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, elevation, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0.35, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.7, d]} />
          <meshStandardMaterial color="#6b4c3a" roughness={0.7} />
        </mesh>
        {/* Door lines */}
        <mesh position={[0, 0.35, d / 2 + 0.001]}>
          <boxGeometry args={[0.01, 0.65, 0.002]} />
          <meshStandardMaterial color="#3a2818" roughness={0.6} />
        </mesh>
        {/* Handles */}
        <mesh position={[-0.02, 0.35, d / 2 + 0.01]}>
          <boxGeometry args={[0.015, 0.04, 0.02]} />
          <meshStandardMaterial color="#888" metalness={0.6} roughness={0.2} />
        </mesh>
        <mesh position={[0.02, 0.35, d / 2 + 0.01]}>
          <boxGeometry args={[0.015, 0.04, 0.02]} />
          <meshStandardMaterial color="#888" metalness={0.6} roughness={0.2} />
        </mesh>
      </group>
    </group>
  );
});

/** Wall cabinet — wall-mounted box */
export const WallCabinetModel = memo(function WallCabinetModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const elevation = item.elevation ?? 0.9;
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, elevation, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0.2, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.4, d]} />
          <meshStandardMaterial color="#6b4c3a" roughness={0.7} />
        </mesh>
        <mesh position={[0, 0.2, d / 2 + 0.001]}>
          <boxGeometry args={[0.01, 0.38, 0.002]} />
          <meshStandardMaterial color="#3a2818" roughness={0.6} />
        </mesh>
      </group>
    </group>
  );
});

/** Microwave — small box */
export const MicrowaveModel = memo(function MicrowaveModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const elevation = item.elevation ?? 0;

  return (
    <group position={[wx, elevation, wz]} rotation={[0, facing, 0]}>
      {/* Body */}
      <mesh position={[0, 0.75, 0]} castShadow>
        <boxGeometry args={[0.3, 0.18, 0.2]} />
        <meshStandardMaterial color="#1a1a1e" roughness={0.4} metalness={0.3} />
      </mesh>
      {/* Door window */}
      <mesh position={[0, 0.75, 0.101]}>
        <boxGeometry args={[0.22, 0.12, 0.002]} />
        <meshStandardMaterial
          color="#0a0a1a"
          transparent
          opacity={0.5}
          roughness={0.1}
        />
      </mesh>
      {/* Handle */}
      <mesh position={[0.13, 0.75, 0.105]}>
        <boxGeometry args={[0.015, 0.1, 0.02]} />
        <meshStandardMaterial color="#555" metalness={0.6} roughness={0.2} />
      </mesh>
    </group>
  );
});

/** Sink — box with basin */
export const SinkModel = memo(function SinkModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Counter */}
        <mesh position={[0, 0.7, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.04, d]} />
          <meshStandardMaterial color="#8b5e32" roughness={0.6} />
        </mesh>
        {/* Basin (inset) */}
        <mesh position={[0, 0.68, 0]}>
          <boxGeometry args={[w * 0.7, 0.05, d * 0.7]} />
          <meshStandardMaterial
            color="#2a2a30"
            roughness={0.2}
            metalness={0.4}
          />
        </mesh>
        {/* Faucet */}
        <mesh position={[0, 0.78, -d / 2 + 0.04]} castShadow>
          <cylinderGeometry args={[0.01, 0.01, 0.1, 8]} />
          <meshStandardMaterial color="#888" metalness={0.7} roughness={0.15} />
        </mesh>
        <mesh position={[0, 0.8, -d / 2 + 0.07]}>
          <boxGeometry args={[0.015, 0.015, 0.06]} />
          <meshStandardMaterial color="#888" metalness={0.7} roughness={0.15} />
        </mesh>
        {/* Cabinet below */}
        <mesh position={[0, 0.35, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.66, d]} />
          <meshStandardMaterial color="#6b4c3a" roughness={0.7} />
        </mesh>
      </group>
    </group>
  );
});

/** Dishwasher — box with front panel */
export const DishwasherModel = memo(function DishwasherModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0.35, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.7, d]} />
          <meshStandardMaterial color="#c0c0c0" roughness={0.3} metalness={0.5} />
        </mesh>
        {/* Front panel line */}
        <mesh position={[0, 0.45, d / 2 + 0.001]}>
          <boxGeometry args={[w * 0.8, 0.01, 0.002]} />
          <meshStandardMaterial color="#888" metalness={0.5} />
        </mesh>
        {/* Handle */}
        <mesh position={[0, 0.55, d / 2 + 0.01]}>
          <boxGeometry args={[w * 0.5, 0.02, 0.02]} />
          <meshStandardMaterial color="#333" metalness={0.6} roughness={0.2} />
        </mesh>
        {/* Control panel */}
        <mesh position={[0, 0.65, d / 2 + 0.002]}>
          <boxGeometry args={[w * 0.6, 0.04, 0.004]} />
          <meshStandardMaterial color="#1a1a1e" roughness={0.3} />
        </mesh>
      </group>
    </group>
  );
});

/** Computer — monitor on desk */
export const ComputerModel = memo(function ComputerModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0.6, wz]}>
      {/* Screen */}
      <mesh position={[0, 0.12, 0]} castShadow>
        <boxGeometry args={[0.25, 0.16, 0.015]} />
        <meshStandardMaterial
          color="#0f0f1e"
          emissive="#1a1a3e"
          emissiveIntensity={0.35}
        />
      </mesh>
      {/* Screen bezel */}
      <mesh position={[0, 0.12, -0.002]}>
        <boxGeometry args={[0.26, 0.17, 0.01]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
      </mesh>
      {/* Stand neck */}
      <mesh position={[0, 0.02, -0.01]}>
        <boxGeometry args={[0.03, 0.06, 0.02]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
      {/* Stand base */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.1, 0.01, 0.06]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
    </group>
  );
});

/** Keyboard — flat thin box */
export const KeyboardModel = memo(function KeyboardModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0.625, wz]}>
      <mesh castShadow>
        <boxGeometry args={[0.18, 0.015, 0.07]} />
        <meshStandardMaterial color="#2e333d" roughness={0.85} />
      </mesh>
      {/* Key hint rows */}
      {[0, 0.04, 0.08, 0.12].map((kx, i) => (
        <mesh key={`key-${i}`} position={[-0.05 + kx, 0.008, 0]}>
          <boxGeometry args={[0.03, 0.008, 0.05]} />
          <meshStandardMaterial color="#1a1a20" roughness={0.7} />
        </mesh>
      ))}
    </group>
  );
});

/** Mouse — small flattened sphere */
export const MouseModel = memo(function MouseModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0.625, wz]}>
      <mesh castShadow scale={[1, 0.38, 0.72]}>
        <sphereGeometry args={[0.03, 12, 8]} />
        <meshStandardMaterial color="#d0cecc" roughness={0.6} />
      </mesh>
    </group>
  );
});

/** Clock — circle with hands (wall-mounted) */
export const ClockModel = memo(function ClockModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 1.5, wz]}>
      {/* Clock face */}
      <mesh castShadow>
        <cylinderGeometry args={[0.1, 0.1, 0.015, 24]} />
        <meshStandardMaterial color="#f4f2ee" roughness={0.3} />
      </mesh>
      {/* Rim */}
      <mesh position={[0, -0.008, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.1, 0.012, 8, 24]} />
        <meshStandardMaterial color="#333" metalness={0.5} roughness={0.3} />
      </mesh>
      {/* Hour hand */}
      <mesh position={[0, 0.008, 0.03]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.008, 0.04, 0.004]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
      {/* Minute hand */}
      <mesh position={[0.02, 0.008, 0.03]} rotation={[0, -0.5, 0]}>
        <boxGeometry args={[0.005, 0.06, 0.003]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
    </group>
  );
});

/** Ping pong table — box surface + net */
export const PingPongModel = memo(function PingPongModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Surface */}
        <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.03, d]} />
          <meshStandardMaterial color="#1a5276" roughness={0.4} />
        </mesh>
        {/* Net */}
        <mesh position={[0, 0.67, 0]}>
          <boxGeometry args={[w, 0.08, 0.005]} />
          <meshStandardMaterial color="#ffffff" transparent opacity={0.7} />
        </mesh>
        {/* Legs */}
        {[
          [-w / 2 + 0.03, -d / 2 + 0.03],
          [w / 2 - 0.03, -d / 2 + 0.03],
          [-w / 2 + 0.03, d / 2 - 0.03],
          [w / 2 - 0.03, d / 2 - 0.03],
        ].map(([lx, lz], i) => (
          <mesh key={`pp-leg-${i}`} position={[lx, 0.3, lz]} castShadow>
            <boxGeometry args={[0.04, 0.6, 0.04]} />
            <meshStandardMaterial color="#1a1a1a" metalness={0.4} roughness={0.3} />
          </mesh>
        ))}
      </group>
    </group>
  );
});

/** ATM — tall narrow box with screen */
export const AtmModel = memo(function AtmModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Body */}
        <mesh position={[0, 0.85, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 1.7, d]} />
          <meshStandardMaterial color="#2a4a6a" roughness={0.4} metalness={0.3} />
        </mesh>
        {/* Screen */}
        <mesh position={[0, 1.2, d / 2 + 0.001]}>
          <boxGeometry args={[w * 0.7, 0.2, 0.005]} />
          <meshStandardMaterial
            color="#0a0a1a"
            emissive="#1a3a5a"
            emissiveIntensity={0.3}
          />
        </mesh>
        {/* Card slot */}
        <mesh position={[0, 0.7, d / 2 + 0.002]}>
          <boxGeometry args={[w * 0.4, 0.02, 0.004]} />
          <meshStandardMaterial color="#111" />
        </mesh>
        {/* Cash slot */}
        <mesh position={[0, 0.5, d / 2 + 0.002]}>
          <boxGeometry args={[w * 0.5, 0.06, 0.004]} />
          <meshStandardMaterial color="#111" />
        </mesh>
      </group>
    </group>
  );
});

/** Phone booth — tall enclosed box */
export const PhoneBoothModel = memo(function PhoneBoothModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Frame */}
        <mesh position={[0, 1.0, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 2.0, d]} />
          <meshStandardMaterial color="#8b4a3a" roughness={0.5} />
        </mesh>
        {/* Glass */}
        <mesh position={[0, 1.1, d / 2 + 0.001]}>
          <boxGeometry args={[w * 0.85, 1.6, 0.005]} />
          <meshStandardMaterial
            color="#a0c0d0"
            transparent
            opacity={0.3}
            roughness={0.1}
          />
        </mesh>
        {/* Roof */}
        <mesh position={[0, 2.05, 0]} castShadow>
          <boxGeometry args={[w * 1.1, 0.06, d * 1.1]} />
          <meshStandardMaterial color="#5a3020" roughness={0.6} />
        </mesh>
      </group>
    </group>
  );
});

/** SMS booth — small booth box */
export const SmsBoothModel = memo(function SmsBoothModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Body */}
        <mesh position={[0, 0.75, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 1.5, d]} />
          <meshStandardMaterial color="#3a4a6a" roughness={0.4} metalness={0.3} />
        </mesh>
        {/* Screen */}
        <mesh position={[0, 1.0, d / 2 + 0.001]}>
          <boxGeometry args={[w * 0.6, 0.3, 0.005]} />
          <meshStandardMaterial
            color="#0a0a1a"
            emissive="#2a4a8a"
            emissiveIntensity={0.4}
          />
        </mesh>
        {/* Roof */}
        <mesh position={[0, 1.55, 0]} castShadow>
          <boxGeometry args={[w * 1.1, 0.05, d * 1.1]} />
          <meshStandardMaterial color="#2a3a5a" roughness={0.5} />
        </mesh>
      </group>
    </group>
  );
});

/** Server rack — tall narrow cabinet */
export const ServerRackModel = memo(function ServerRackModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Frame */}
        <mesh position={[0, 0.9, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 1.8, d]} />
          <meshStandardMaterial color="#1a1a1e" roughness={0.4} metalness={0.3} />
        </mesh>
        {/* Server units with LED indicators */}
        {[0.2, 0.5, 0.8, 1.1, 1.4].map((uy, i) => (
          <group key={`rack-unit-${i}`}>
            <mesh position={[0, uy, d / 2 + 0.001]}>
              <boxGeometry args={[w * 0.9, 0.12, 0.004]} />
              <meshStandardMaterial color="#2a2a2e" roughness={0.3} />
            </mesh>
            {/* LED */}
            <mesh position={[w * 0.35, uy, d / 2 + 0.003]}>
              <sphereGeometry args={[0.006, 6, 6]} />
              <meshStandardMaterial
                color="#22c55e"
                emissive="#22c55e"
                emissiveIntensity={0.8}
              />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  );
});

/** Server terminal — small terminal on stand */
export const ServerTerminalModel = memo(function ServerTerminalModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Stand */}
      <mesh position={[0, 0.3, 0]} castShadow>
        <boxGeometry args={[0.2, 0.6, 0.15]} />
        <meshStandardMaterial color="#2a2a2e" roughness={0.5} />
      </mesh>
      {/* Screen */}
      <mesh position={[0, 0.75, 0]} castShadow>
        <boxGeometry args={[0.25, 0.18, 0.03]} />
        <meshStandardMaterial
          color="#0f0f1e"
          emissive="#1a4a2a"
          emissiveIntensity={0.4}
        />
      </mesh>
      {/* Base */}
      <mesh position={[0, 0.02, 0]}>
        <boxGeometry args={[0.25, 0.04, 0.2]} />
        <meshStandardMaterial color="#1a1a1e" />
      </mesh>
    </group>
  );
});

/** Wall segment — box geometry */
export const WallModel = memo(function WallModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]}>
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0.75, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 1.5, Math.max(d, 0.04)]} />
          <meshStandardMaterial color="#c0b8a8" roughness={0.85} />
        </mesh>
      </group>
    </group>
  );
});

/** Door — door frame */
export const DoorModel = memo(function DoorModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Door panel (slightly transparent to indicate passable) */}
        <mesh position={[0, 0.75, 0]} receiveShadow>
          <boxGeometry args={[w, 1.5, Math.max(d, 0.02)]} />
          <meshStandardMaterial
            color="#8b6a4a"
            transparent
            opacity={0.15}
            roughness={0.5}
          />
        </mesh>
        {/* Frame top */}
        <mesh position={[0, 1.55, 0]} castShadow>
          <boxGeometry args={[w + 0.04, 0.06, d + 0.04]} />
          <meshStandardMaterial color="#5c3520" roughness={0.7} />
        </mesh>
        {/* Frame sides */}
        <mesh position={[-w / 2 - 0.02, 0.75, 0]} castShadow>
          <boxGeometry args={[0.04, 1.5, d + 0.04]} />
          <meshStandardMaterial color="#5c3520" roughness={0.7} />
        </mesh>
        <mesh position={[w / 2 + 0.02, 0.75, 0]} castShadow>
          <boxGeometry args={[0.04, 1.5, d + 0.04]} />
          <meshStandardMaterial color="#5c3520" roughness={0.7} />
        </mesh>
      </group>
    </group>
  );
});

/** Treadmill — gym equipment: base + handles */
export const TreadmillModel = memo(function TreadmillModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Base / belt */}
      <mesh position={[0, 0.15, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.3, 0.06, 0.6]} />
        <meshStandardMaterial color="#1a1a1e" roughness={0.4} />
      </mesh>
      {/* Belt surface */}
      <mesh position={[0, 0.181, 0]}>
        <boxGeometry args={[0.24, 0.001, 0.5]} />
        <meshStandardMaterial color="#2a2a2e" roughness={0.6} />
      </mesh>
      {/* Front riser */}
      <mesh position={[0, 0.3, -0.25]} castShadow>
        <boxGeometry args={[0.3, 0.3, 0.04]} />
        <meshStandardMaterial color="#2a2a2e" roughness={0.4} />
      </mesh>
      {/* Console */}
      <mesh position={[0, 0.55, -0.27]} castShadow>
        <boxGeometry args={[0.2, 0.1, 0.04]} />
        <meshStandardMaterial
          color="#1a1a1e"
          emissive="#1a3a5a"
          emissiveIntensity={0.2}
        />
      </mesh>
      {/* Side rails */}
      <mesh position={[-0.16, 0.16, 0]}>
        <boxGeometry args={[0.02, 0.03, 0.6]} />
        <meshStandardMaterial color="#333" roughness={0.5} />
      </mesh>
      <mesh position={[0.16, 0.16, 0]}>
        <boxGeometry args={[0.02, 0.03, 0.6]} />
        <meshStandardMaterial color="#333" roughness={0.5} />
      </mesh>
    </group>
  );
});

/** Weight bench — flat box on legs */
export const WeightBenchModel = memo(function WeightBenchModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Bench surface */}
      <mesh position={[0, 0.35, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.25, 0.05, 0.7]} />
        <meshStandardMaterial color="#1a1a1e" roughness={0.5} />
      </mesh>
      {/* Padding */}
      <mesh position={[0, 0.375, 0]}>
        <boxGeometry args={[0.22, 0.02, 0.65]} />
        <meshStandardMaterial color="#3a3a3e" roughness={0.7} />
      </mesh>
      {/* Legs */}
      {[
        [-0.1, -0.3],
        [0.1, -0.3],
        [-0.1, 0.3],
        [0.1, 0.3],
      ].map(([lx, lz], i) => (
        <mesh key={`wb-leg-${i}`} position={[lx, 0.17, lz]} castShadow>
          <boxGeometry args={[0.03, 0.35, 0.03]} />
          <meshStandardMaterial color="#2a2a2e" metalness={0.4} roughness={0.3} />
        </mesh>
      ))}
      {/* Uprights for bar */}
      <mesh position={[0, 0.4, -0.35]} castShadow>
        <boxGeometry args={[0.03, 0.3, 0.03]} />
        <meshStandardMaterial color="#2a2a2e" metalness={0.4} roughness={0.3} />
      </mesh>
      <mesh position={[-0.12, 0.55, -0.35]}>
        <cylinderGeometry args={[0.008, 0.008, 0.25, 8]} />
        <meshStandardMaterial color="#888" metalness={0.6} roughness={0.2} />
      </mesh>
    </group>
  );
});

/** Dumbbell rack — shelf with small dumbbells */
export const DumbbellRackModel = memo(function DumbbellRackModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Frame */}
        <mesh position={[0, 0.25, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.5, d]} />
          <meshStandardMaterial color="#2a2a2e" roughness={0.4} metalness={0.3} />
        </mesh>
        {/* Shelves with dumbbells */}
        {[0.15, 0.35].map((sy, shelfIdx) =>
          [-w * 0.3, 0, w * 0.3].map((dx, i) => (
            <group key={`db-${shelfIdx}-${i}`} position={[dx, sy, d / 2 + 0.01]}>
              {/* Handle */}
              <mesh rotation={[0, 0, Math.PI / 2]} castShadow>
                <cylinderGeometry args={[0.008, 0.008, 0.08, 8]} />
                <meshStandardMaterial color="#555" metalness={0.6} roughness={0.2} />
              </mesh>
              {/* Weights */}
              <mesh position={[-0.05, 0, 0]} castShadow>
                <cylinderGeometry args={[0.02, 0.02, 0.02, 12]} />
                <meshStandardMaterial color="#1a1a1a" metalness={0.4} />
              </mesh>
              <mesh position={[0.05, 0, 0]} castShadow>
                <cylinderGeometry args={[0.02, 0.02, 0.02, 12]} />
                <meshStandardMaterial color="#1a1a1a" metalness={0.4} />
              </mesh>
            </group>
          )),
        )}
      </group>
    </group>
  );
});

/** Rowing machine — base with rail */
export const RowingMachineModel = memo(function RowingMachineModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Rail */}
      <mesh position={[0, 0.2, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.06, 0.04, 0.8]} />
        <meshStandardMaterial color="#2a2a2e" metalness={0.4} roughness={0.3} />
      </mesh>
      {/* Seat */}
      <mesh position={[0, 0.25, 0.1]} castShadow>
        <boxGeometry args={[0.18, 0.04, 0.12]} />
        <meshStandardMaterial color="#1a1a1e" roughness={0.6} />
      </mesh>
      {/* Front housing / fan */}
      <mesh position={[0, 0.25, -0.35]} castShadow>
        <cylinderGeometry args={[0.1, 0.1, 0.1, 16]} />
        <meshStandardMaterial color="#1a1a1e" roughness={0.4} />
      </mesh>
      {/* Foot plates */}
      <mesh position={[0, 0.1, -0.25]}>
        <boxGeometry args={[0.2, 0.02, 0.1]} />
        <meshStandardMaterial color="#333" roughness={0.5} />
      </mesh>
    </group>
  );
});

/** Kettlebell rack — shelf with spheres */
export const KettlebellRackModel = memo(function KettlebellRackModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Frame */}
        <mesh position={[0, 0.25, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.5, d]} />
          <meshStandardMaterial color="#2a2a2e" roughness={0.4} metalness={0.3} />
        </mesh>
        {/* Kettlebells */}
        {[-w * 0.3, 0, w * 0.3].map((kx, i) => (
          <group key={`kb-${i}`} position={[kx, 0.45, d / 2 + 0.01]}>
            <mesh castShadow>
              <sphereGeometry args={[0.03, 12, 10]} />
              <meshStandardMaterial color="#1a1a1a" metalness={0.4} roughness={0.3} />
            </mesh>
            {/* Handle */}
            <mesh position={[0, 0.035, 0]} rotation={[0, 0, 0]}>
              <torusGeometry args={[0.015, 0.005, 6, 12]} />
              <meshStandardMaterial color="#555" metalness={0.6} roughness={0.2} />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  );
});

/** Exercise bike — base with seat + pedals */
export const ExerciseBikeModel = memo(function ExerciseBikeModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Main body */}
      <mesh position={[0, 0.25, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.15, 0.3, 0.4]} />
        <meshStandardMaterial color="#2a2a2e" roughness={0.4} metalness={0.3} />
      </mesh>
      {/* Flywheel */}
      <mesh position={[0, 0.15, -0.15]} castShadow>
        <cylinderGeometry args={[0.08, 0.08, 0.03, 16]} />
        <meshStandardMaterial color="#1a1a1e" metalness={0.5} roughness={0.2} />
      </mesh>
      {/* Seat post */}
      <mesh position={[0, 0.5, 0.15]} castShadow>
        <cylinderGeometry args={[0.015, 0.015, 0.4, 8]} />
        <meshStandardMaterial color="#333" metalness={0.5} roughness={0.3} />
      </mesh>
      {/* Seat */}
      <mesh position={[0, 0.72, 0.15]} castShadow>
        <boxGeometry args={[0.12, 0.04, 0.1]} />
        <meshStandardMaterial color="#1a1a1e" roughness={0.6} />
      </mesh>
      {/* Handlebars */}
      <mesh position={[0, 0.7, -0.15]} castShadow>
        <cylinderGeometry args={[0.015, 0.015, 0.3, 8]} />
        <meshStandardMaterial color="#333" metalness={0.5} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0.85, -0.15]}>
        <boxGeometry args={[0.25, 0.02, 0.02]} />
        <meshStandardMaterial color="#1a1a1e" roughness={0.5} />
      </mesh>
      {/* Pedals */}
      <mesh position={[0.06, 0.12, 0]}>
        <boxGeometry args={[0.04, 0.01, 0.06]} />
        <meshStandardMaterial color="#555" />
      </mesh>
      <mesh position={[-0.06, 0.12, 0]}>
        <boxGeometry args={[0.04, 0.01, 0.06]} />
        <meshStandardMaterial color="#555" />
      </mesh>
    </group>
  );
});

/** Punching bag — hanging cylinder */
export const PunchingBagModel = memo(function PunchingBagModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Chain */}
      <mesh position={[0, 1.95, 0]}>
        <cylinderGeometry args={[0.005, 0.005, 0.2, 6]} />
        <meshStandardMaterial color="#555" metalness={0.7} roughness={0.2} />
      </mesh>
      {/* Mount bracket */}
      <mesh position={[0, 2.1, 0]}>
        <boxGeometry args={[0.08, 0.02, 0.08]} />
        <meshStandardMaterial color="#333" metalness={0.6} />
      </mesh>
      {/* Bag */}
      <mesh position={[0, 1.3, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.12, 0.1, 0.9, 16]} />
        <meshStandardMaterial color="#8b2a2a" roughness={0.85} />
      </mesh>
      {/* Bag top cap */}
      <mesh position={[0, 1.76, 0]}>
        <cylinderGeometry args={[0.12, 0.12, 0.02, 16]} />
        <meshStandardMaterial color="#5a1a1a" roughness={0.7} />
      </mesh>
    </group>
  );
});

/** Yoga mat — flat thin colored plane */
export const YogaMatModel = memo(function YogaMatModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;
  const color = item.color ?? "#0f766e";

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0.005, 0]} receiveShadow>
          <boxGeometry args={[w, 0.01, d]} />
          <meshStandardMaterial color={color} roughness={0.9} />
        </mesh>
      </group>
    </group>
  );
});

/** Easel — tripod stand */
export const EaselModel = memo(function EaselModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Tripod legs */}
      <mesh position={[0, 0.75, 0.05]} rotation={[0.15, 0, 0]} castShadow>
        <cylinderGeometry args={[0.012, 0.012, 1.5, 8]} />
        <meshStandardMaterial color="#6b4c3a" roughness={0.8} />
      </mesh>
      <mesh position={[-0.08, 0.75, -0.05]} rotation={[-0.15, -0.3, 0]} castShadow>
        <cylinderGeometry args={[0.012, 0.012, 1.5, 8]} />
        <meshStandardMaterial color="#6b4c3a" roughness={0.8} />
      </mesh>
      <mesh position={[0.08, 0.75, -0.05]} rotation={[-0.15, 0.3, 0]} castShadow>
        <cylinderGeometry args={[0.012, 0.012, 1.5, 8]} />
        <meshStandardMaterial color="#6b4c3a" roughness={0.8} />
      </mesh>
      {/* Crossbar */}
      <mesh position={[0, 0.6, 0]}>
        <boxGeometry args={[0.2, 0.02, 0.02]} />
        <meshStandardMaterial color="#5c3520" roughness={0.7} />
      </mesh>
      {/* Canvas */}
      <mesh position={[0, 0.9, 0.02]} castShadow>
        <boxGeometry args={[0.3, 0.4, 0.02]} />
        <meshStandardMaterial color="#f0e8d8" roughness={0.9} />
      </mesh>
    </group>
  );
});

/** QA terminal — terminal desk */
export const QaTerminalModel = memo(function QaTerminalModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Desk surface */}
        <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.04, d]} />
          <meshStandardMaterial color="#3a3a3e" roughness={0.5} />
        </mesh>
        {/* Legs */}
        {[
          [-w / 2 + 0.03, -d / 2 + 0.03],
          [w / 2 - 0.03, -d / 2 + 0.03],
          [-w / 2 + 0.03, d / 2 - 0.03],
          [w / 2 - 0.03, d / 2 - 0.03],
        ].map(([lx, lz], i) => (
          <mesh key={`qt-leg-${i}`} position={[lx, 0.3, lz]} castShadow>
            <boxGeometry args={[0.04, 0.6, 0.04]} />
            <meshStandardMaterial color="#2a2a2e" />
          </mesh>
        ))}
        {/* Monitor */}
        <mesh position={[0, 0.75, -d / 2 + 0.06]} castShadow>
          <boxGeometry args={[w * 0.6, 0.18, 0.02]} />
          <meshStandardMaterial
            color="#0f0f1e"
            emissive="#1a4a2a"
            emissiveIntensity={0.35}
          />
        </mesh>
        {/* Stand */}
        <mesh position={[0, 0.65, -d / 2 + 0.05]}>
          <boxGeometry args={[0.04, 0.1, 0.04]} />
          <meshStandardMaterial color="#1a1a1a" />
        </mesh>
      </group>
    </group>
  );
});

/** Device rack — equipment rack */
export const DeviceRackModel = memo(function DeviceRackModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Frame */}
        <mesh position={[0, 0.75, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 1.5, d]} />
          <meshStandardMaterial color="#2a2a2e" roughness={0.4} metalness={0.3} />
        </mesh>
        {/* Device slots */}
        {[0.4, 0.65, 0.9, 1.15].map((sy, i) => (
          <mesh key={`dr-slot-${i}`} position={[0, sy, d / 2 + 0.002]}>
            <boxGeometry args={[w * 0.85, 0.1, 0.004]} />
            <meshStandardMaterial
              color="#0f0f1e"
              emissive="#1a3a5a"
              emissiveIntensity={0.2}
            />
          </mesh>
        ))}
      </group>
    </group>
  );
});

/** Test bench — workbench */
export const RugModel = memo(function RugModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const w = (item.w ?? 200) * SCALE;
  const d = (item.h ?? 150) * SCALE;
  return (
    <group position={[wx, 0, wz]} rotation={[-Math.PI / 2, 0, 0]}>
      <mesh position={[0, 0.002, 0]} receiveShadow>
        <planeGeometry args={[w, d]} />
        <meshStandardMaterial
          color="#8B4513"
          roughness={0.95}
          metalness={0.0}
          side={2}
        />
      </mesh>
      <mesh position={[0, 0.003, 0]}>
        <ringGeometry args={[Math.min(w, d) * 0.3, Math.min(w, d) * 0.45, 32]} />
        <meshStandardMaterial color="#D4A574" roughness={0.9} side={2} />
      </mesh>
    </group>
  );
});

export const TvModel = memo(function TvModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const w = (item.w ?? 120) * SCALE;
  const h = 0.8;
  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* TV screen */}
      <mesh position={[0, h + 0.4, 0]} castShadow>
        <boxGeometry args={[w, h * 0.6, 0.04]} />
        <meshStandardMaterial
          color="#0a0a0a"
          emissive="#1a1a2e"
          emissiveIntensity={0.15}
        />
      </mesh>
      {/* Screen glow */}
      <mesh position={[0, h + 0.4, 0.025]}>
        <planeGeometry args={[w * 0.92, h * 0.52]} />
        <meshBasicMaterial color="#16204a" />
      </mesh>
      {/* Stand */}
      <mesh position={[0, 0.15, 0]} castShadow>
        <boxGeometry args={[w * 0.3, 0.3, 0.06]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.8} />
      </mesh>
    </group>
  );
});

/** Dual monitor setup — two screens side by side on a stand */
export const DualMonitorModel = memo(function DualMonitorModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);

  return (
    <group position={[wx, 0.6, wz]} rotation={[0, facing, 0]}>
      {/* Left screen */}
      <mesh position={[-0.14, 0.12, 0]} castShadow>
        <boxGeometry args={[0.24, 0.16, 0.015]} />
        <meshStandardMaterial color="#0f0f1e" emissive="#1a2a4e" emissiveIntensity={0.3} />
      </mesh>
      <mesh position={[-0.14, 0.12, -0.002]}>
        <boxGeometry args={[0.25, 0.17, 0.01]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
      </mesh>
      {/* Right screen */}
      <mesh position={[0.14, 0.12, 0]} castShadow>
        <boxGeometry args={[0.24, 0.16, 0.015]} />
        <meshStandardMaterial color="#0f0f1e" emissive="#2a1a4e" emissiveIntensity={0.3} />
      </mesh>
      <mesh position={[0.14, 0.12, -0.002]}>
        <boxGeometry args={[0.25, 0.17, 0.01]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
      </mesh>
      {/* Center stand neck */}
      <mesh position={[0, 0.02, -0.01]}>
        <boxGeometry args={[0.03, 0.06, 0.02]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
      {/* Stand base */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.12, 0.01, 0.07]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
    </group>
  );
});

/** Coffee bar — counter with espresso machine, mugs, and grinder */
export const CoffeeBarModel = memo(function CoffeeBarModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Counter base */}
        <mesh position={[0, 0.45, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.9, d]} />
          <meshStandardMaterial color="#3a2a1a" roughness={0.6} />
        </mesh>
        {/* Counter top */}
        <mesh position={[0, 0.91, 0]} castShadow>
          <boxGeometry args={[w + 0.02, 0.03, d + 0.02]} />
          <meshStandardMaterial color="#1a1a1a" roughness={0.3} metalness={0.2} />
        </mesh>
        {/* Espresso machine */}
        <mesh position={[-w * 0.25, 1.05, 0]} castShadow>
          <boxGeometry args={[0.12, 0.22, 0.1]} />
          <meshStandardMaterial color="#c0c0c0" roughness={0.3} metalness={0.5} />
        </mesh>
        <mesh position={[-w * 0.25, 1.18, 0]}>
          <boxGeometry args={[0.08, 0.04, 0.06]} />
          <meshStandardMaterial color="#1a1a1a" />
        </mesh>
        {/* Grinder */}
        <mesh position={[-w * 0.05, 1.0, 0]} castShadow>
          <boxGeometry args={[0.06, 0.16, 0.06]} />
          <meshStandardMaterial color="#2a2a2e" roughness={0.4} />
        </mesh>
        {/* Mug row */}
        {[-0.12, -0.04, 0.04, 0.12].map((mx, i) => (
          <mesh key={`mug-${i}`} position={[w * 0.15 + mx, 0.97, 0]}>
            <cylinderGeometry args={[0.025, 0.02, 0.04, 8]} />
            <meshStandardMaterial color={i % 2 === 0 ? "#e5e5e5" : "#d4a574"} roughness={0.5} />
          </mesh>
        ))}
      </group>
    </group>
  );
});

/** Snack shelf — open wall shelving with colorful snack items */
export const SnackShelfModel = memo(function SnackShelfModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  const shelfYs = [1.0, 1.25, 1.5];
  const snackColors = ["#e63946", "#f4a261", "#2a9d8f", "#e9c46a", "#a06cd5", "#ef476f"];

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Back panel */}
        <mesh position={[0, 1.25, -d / 2 + 0.01]}>
          <boxGeometry args={[w, 0.65, 0.01]} />
          <meshStandardMaterial color="#2a2a1e" roughness={0.7} />
        </mesh>
        {/* Shelves */}
        {shelfYs.map((sy, i) => (
          <mesh key={`shelf-${i}`} position={[0, sy, 0]}>
            <boxGeometry args={[w, 0.02, d]} />
            <meshStandardMaterial color="#3a2a1a" roughness={0.6} />
          </mesh>
        ))}
        {/* Snack items on shelves */}
        {shelfYs.map((sy, si) =>
          [-0.3, -0.1, 0.1, 0.3].map((sx, i) => (
            <mesh key={`snack-${si}-${i}`} position={[sx, sy + 0.04, 0]}>
              <boxGeometry args={[0.04, 0.06, 0.04]} />
              <meshStandardMaterial
                color={snackColors[(si * 4 + i) % snackColors.length]}
                roughness={0.5}
              />
            </mesh>
          ))
        )}
      </group>
    </group>
  );
});

/** Wall poster / framed art — flat panel on wall */
export const PosterModel = memo(function PosterModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const posterColors = [
    ["#1a1a2e", "#16213e", "#0f3460"],
    ["#2d1b00", "#5c3d00", "#8b5a00"],
    ["#1a2e1a", "#2e4e2e", "#4e6e4e"],
  ];
  const variant = (item.x + item.y) % 3;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      {/* Frame */}
      <mesh position={[0, 1.3, 0]} castShadow>
        <boxGeometry args={[0.3, 0.4, 0.02]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.4} />
      </mesh>
      {/* Poster art */}
      <mesh position={[0, 1.3, 0.012]}>
        <boxGeometry args={[0.26, 0.36, 0.002]} />
        <meshStandardMaterial
          color={posterColors[variant][0]}
          emissive={posterColors[variant][1]}
          emissiveIntensity={0.15}
          roughness={0.5}
        />
      </mesh>
      {/* Accent stripe */}
      <mesh position={[0, 1.15, 0.014]}>
        <boxGeometry args={[0.2, 0.04, 0.001]} />
        <meshStandardMaterial
          color={posterColors[variant][2]}
          emissive={posterColors[variant][2]}
          emissiveIntensity={0.2}
        />
      </mesh>
    </group>
  );
});

/** Trophy — golden cup on a pedestal */
export const TrophyModel = memo(function TrophyModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);

  return (
    <group position={[wx, 0, wz]}>
      {/* Pedestal */}
      <mesh position={[0, 0.3, 0]} castShadow>
        <boxGeometry args={[0.08, 0.6, 0.08]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
      </mesh>
      {/* Cup body */}
      <mesh position={[0, 0.7, 0]} castShadow>
        <cylinderGeometry args={[0.05, 0.04, 0.12, 12]} />
        <meshStandardMaterial color="#ffd700" roughness={0.2} metalness={0.8} />
      </mesh>
      {/* Cup rim */}
      <mesh position={[0, 0.77, 0]}>
        <cylinderGeometry args={[0.06, 0.05, 0.02, 12]} />
        <meshStandardMaterial color="#ffd700" roughness={0.2} metalness={0.8} />
      </mesh>
      {/* Handles */}
      <mesh position={[-0.07, 0.72, 0]}>
        <torusGeometry args={[0.03, 0.006, 6, 12, Math.PI]} />
        <meshStandardMaterial color="#ffd700" roughness={0.2} metalness={0.8} />
      </mesh>
      <mesh position={[0.07, 0.72, 0]} rotation={[0, 0, Math.PI]}>
        <torusGeometry args={[0.03, 0.006, 6, 12, Math.PI]} />
        <meshStandardMaterial color="#ffd700" roughness={0.2} metalness={0.8} />
      </mesh>
    </group>
  );
});

/** Rubik's cube — small colorful cube on desk */
export const RubiksCubeModel = memo(function RubiksCubeModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const colors = [
    "#e63946", "#f4a261", "#2a9d8f",
    "#e9c46a", "#264653", "#f1faee",
  ];

  return (
    <group position={[wx, 0.63, wz]} rotation={[0, facing, 0]}>
      {/* Core */}
      <mesh castShadow>
        <boxGeometry args={[0.06, 0.06, 0.06]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.8} />
      </mesh>
      {/* Face stickers */}
      {colors.map((c, i) => {
        const face = [
          [0, 0, 0.031, 0, 0, 0],
          [0, 0, -0.031, 0, Math.PI, 0],
          [0.031, 0, 0, 0, Math.PI / 2, 0],
          [-0.031, 0, 0, 0, -Math.PI / 2, 0],
          [0, 0.031, 0, -Math.PI / 2, 0, 0],
          [0, -0.031, 0, Math.PI / 2, 0, 0],
        ][i];
        return (
          <mesh
            key={`sticker-${i}`}
            position={face.slice(0, 3) as [number, number, number]}
            rotation={face.slice(3) as [number, number, number]}
          >
            <planeGeometry args={[0.04, 0.04]} />
            <meshStandardMaterial color={c} roughness={0.4} />
          </mesh>
        );
      })}
    </group>
  );
});

/** Pendant lamp — hanging lamp for over-table lighting */
export const PendantLampModel = memo(function PendantLampModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const cord = item.elevation ?? 1.8;

  return (
    <group position={[wx, 0, wz]}>
      {/* Cord */}
      <mesh position={[0, (cord + 2.4) / 2, 0]}>
        <cylinderGeometry args={[0.003, 0.003, cord + 2.4, 4]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
      {/* Shade */}
      <mesh position={[0, cord, 0]} castShadow>
        <coneGeometry args={[0.08, 0.1, 8]} />
        <meshStandardMaterial color="#2a2a2e" roughness={0.5} />
      </mesh>
      {/* Bulb glow */}
      <mesh position={[0, cord - 0.03, 0]}>
        <sphereGeometry args={[0.025, 8, 8]} />
        <meshStandardMaterial
          color="#ffd700"
          emissive="#ffaa00"
          emissiveIntensity={0.6}
        />
      </mesh>
      {/* Warm light pool */}
      <pointLight
        position={[0, cord - 0.05, 0]}
        intensity={0.3}
        distance={2}
        color="#ffcc77"
      />
    </group>
  );
});

export const TestBenchModel = memo(function TestBenchModel({
  item,
}: {
  item: FurnitureItem;
}) {
  const [wx, , wz] = toWorld(item.x, item.y);
  const facing = getItemRotationRadians(item);
  const { width, height } = getItemBaseSize(item);
  const w = width * SCALE;
  const d = height * SCALE;

  return (
    <group position={[wx, 0, wz]} rotation={[0, facing, 0]}>
      <group position={[0, 0, 0]}>
        {/* Surface */}
        <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.04, d]} />
          <meshStandardMaterial color="#4a4a3a" roughness={0.6} />
        </mesh>
        {/* Legs */}
        {[
          [-w / 2 + 0.03, -d / 2 + 0.03],
          [w / 2 - 0.03, -d / 2 + 0.03],
          [-w / 2 + 0.03, d / 2 - 0.03],
          [w / 2 - 0.03, d / 2 - 0.03],
        ].map(([lx, lz], i) => (
          <mesh key={`tb-leg-${i}`} position={[lx, 0.3, lz]} castShadow>
            <boxGeometry args={[0.04, 0.6, 0.04]} />
            <meshStandardMaterial color="#2a2a2e" />
          </mesh>
        ))}
        {/* Test equipment on surface */}
        <mesh position={[0, 0.68, -d / 2 + 0.08]} castShadow>
          <boxGeometry args={[w * 0.5, 0.12, 0.08]} />
          <meshStandardMaterial
            color="#1a1a1e"
            emissive="#1a4a2a"
            emissiveIntensity={0.25}
          />
        </mesh>
        {/* Shelving below */}
        <mesh position={[0, 0.3, 0]}>
          <boxGeometry args={[w * 0.9, 0.02, d * 0.9]} />
          <meshStandardMaterial color="#3a3a2e" roughness={0.7} />
        </mesh>
      </group>
    </group>
  );
});
