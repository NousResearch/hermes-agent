/**
 * Claw3D camera & lighting — adapted from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 * Copyright (c) 2026 zhuohoudeputao (Hermes integration)
 *
 * Day/night cycle lighting and camera controller.
 * Simplified from Claw3D — removed follow-cam and orbit animation,
 * kept the day-night cycle for visual ambiance.
 */

import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";
import { WORLD_H, WORLD_W } from "./constants";

const DAY_NIGHT_PERIOD = 300;

const DAY_NIGHT_POSITIONS = [0, 0.2, 0.45, 0.65, 0.8, 0.95];

const DAY_NIGHT_KEYFRAMES = [
  { ambient: "#c8a870", sun: "#ffe8b0", sunIntensity: 0.8, ambientIntensity: 0.55 },
  { ambient: "#c8d0e0", sun: "#f0f4ff", sunIntensity: 1.3, ambientIntensity: 0.75 },
  { ambient: "#c8d0e0", sun: "#f0f4ff", sunIntensity: 1.3, ambientIntensity: 0.75 },
  { ambient: "#c87840", sun: "#ff9050", sunIntensity: 0.9, ambientIntensity: 0.5 },
  { ambient: "#1a2040", sun: "#2040a0", sunIntensity: 0.3, ambientIntensity: 0.25 },
  { ambient: "#101828", sun: "#182038", sunIntensity: 0.2, ambientIntensity: 0.2 },
];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

const lerpColor = (fromColor: string, toColor: string, t: number) => {
  const parse = (color: string) => {
    const value = parseInt(color.slice(1), 16);
    return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff];
  };
  const [fromR, fromG, fromB] = parse(fromColor);
  const [toR, toG, toB] = parse(toColor);
  const red = Math.round(lerp(fromR, toR, t));
  const green = Math.round(lerp(fromG, toG, t));
  const blue = Math.round(lerp(fromB, toB, t));
  return `rgb(${red},${green},${blue})`;
};

export function DayNightCycle() {
  const ambientRef = useRef<THREE.AmbientLight>(null);
  const sunRef = useRef<THREE.DirectionalLight>(null);
  const timeRef = useRef(0.25);

  useFrame((_, delta) => {
    timeRef.current = (timeRef.current + delta / DAY_NIGHT_PERIOD) % 1;
    const time = timeRef.current;

    let indexA = 0;
    for (let index = 0; index < DAY_NIGHT_POSITIONS.length - 1; index += 1) {
      if (time >= DAY_NIGHT_POSITIONS[index] && time < DAY_NIGHT_POSITIONS[index + 1]) {
        indexA = index;
        break;
      }
      if (time >= DAY_NIGHT_POSITIONS[DAY_NIGHT_POSITIONS.length - 1]) {
        indexA = DAY_NIGHT_POSITIONS.length - 1;
      }
    }

    const indexB = (indexA + 1) % DAY_NIGHT_KEYFRAMES.length;
    const positionA = DAY_NIGHT_POSITIONS[indexA];
    const positionB = indexB === 0 ? 1 : DAY_NIGHT_POSITIONS[indexB];
    const span = positionB - positionA;
    const localT = span > 0 ? (time - positionA) / span : 0;
    const keyframeA = DAY_NIGHT_KEYFRAMES[indexA];
    const keyframeB = DAY_NIGHT_KEYFRAMES[indexB];

    if (ambientRef.current) {
      ambientRef.current.color.set(
        lerpColor(keyframeA.ambient, keyframeB.ambient, localT),
      );
      ambientRef.current.intensity = lerp(
        keyframeA.ambientIntensity,
        keyframeB.ambientIntensity,
        localT,
      );
    }

    if (sunRef.current) {
      sunRef.current.color.set(lerpColor(keyframeA.sun, keyframeB.sun, localT));
      sunRef.current.intensity = lerp(
        keyframeA.sunIntensity,
        keyframeB.sunIntensity,
        localT,
      );
    }
  });

  return (
    <>
      <ambientLight ref={ambientRef} intensity={0.75} color="#c8d0e0" />
      <directionalLight
        ref={sunRef}
        position={[8, 14, 6]}
        intensity={1.3}
        color="#f0f4ff"
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-bias={-0.0002}
        shadow-normalBias={0.02}
        shadow-camera-left={-WORLD_W * 0.7}
        shadow-camera-right={WORLD_W * 0.7}
        shadow-camera-top={WORLD_H * 0.7}
        shadow-camera-bottom={-WORLD_H * 0.7}
      />
      {/* Fill light for the office interior */}
      <pointLight position={[-3, 3, 2]} intensity={0.3} color="#a0b8d8" />
      <pointLight position={[3, 3, -2]} intensity={0.2} color="#d8b8a0" />
    </>
  );
}
