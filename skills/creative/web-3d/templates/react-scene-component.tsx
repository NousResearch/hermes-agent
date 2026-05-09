import { Canvas } from "@react-three/fiber";

function SceneContents() {
  return (
    <mesh rotation={[0.4, 0.4, 0]}>
      <torusKnotGeometry args={[0.75, 0.24, 180, 24]} />
      <meshStandardMaterial color="#d9e7ff" roughness={0.28} metalness={0.2} />
    </mesh>
  );
}

export function SceneHero() {
  return (
    <div style={{ width: "100%", height: "100%" }}>
      <Canvas dpr={[1, 2]} camera={{ position: [0, 0, 4], fov: 45 }}>
        <color attach="background" args={["#0a0c10"]} />
        <ambientLight intensity={0.65} />
        <directionalLight position={[2, 2, 2]} intensity={1.2} />
        <SceneContents />
      </Canvas>
    </div>
  );
}
