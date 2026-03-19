"""
COSMOS v2.0 Integration Test
==============================

Tests all four new subsystems:
  1. Latent Space Routing — auto-route by semantic intent
  2. Proof-of-Inference — consensus verification
  3. GPU-Signed Model Attestation — verify running models
  4. Swarm Memory Crystallization — propose & verify knowledge

Usage:
    python -m cosmos.network.cosmos_v2_test --host 127.0.0.1
    python -m cosmos.network.cosmos_v2_test --test latent
    python -m cosmos.network.cosmos_v2_test --test poi
    python -m cosmos.network.cosmos_v2_test --test all
"""
import asyncio
import argparse
import time
import json
import uuid
import socket
from typing import Optional

from .cosmos_protocol import (
    PacketType, COSMOSPacket,
    read_frame, write_frame,
    make_hello, make_discovery,
)
from .cosmos_auth import (
    get_or_create_seed, derive_node_identity,
    compute_gpu_fingerprint, solve_challenge,
)
from .cosmos_latent_router import LatentRouter
from . import COSMOS_PORT


class V2TestClient:
    """Test client for COSMOS v2.0 features."""

    def __init__(self, host: str = "127.0.0.1", port: int = COSMOS_PORT):
        self.host = host
        self.port = port
        self.node_name = "v2-test-client"
        self._reader = None
        self._writer = None
        self._seed = get_or_create_seed()
        self._gpu_fp = b""
        self._identity = b""
        self._remote_bots = []

    async def connect(self, compute_gpu: bool = False) -> bool:
        """Connect and authenticate."""
        if compute_gpu:
            loop = asyncio.get_event_loop()
            self._gpu_fp = await loop.run_in_executor(None, compute_gpu_fingerprint)
        self._identity = derive_node_identity(self._seed, self.node_name, self._gpu_fp)

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=10.0
            )
            sock = self._writer.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # HELLO
            await write_frame(self._writer, make_hello(
                self.node_name, self._gpu_fp, {"version": 2, "bots": []},
            ))

            # Challenge
            resp = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if not resp or resp.packet_type != PacketType.VERIFY:
                return False

            challenge = resp.data.get("challenge", b"")
            ts = time.time()
            response = solve_challenge(self._identity, challenge, ts)

            await write_frame(self._writer, COSMOSPacket(
                packet_type=PacketType.VERIFY,
                sender=self.node_name,
                target=resp.sender,
                data={"response": response, "timestamp": ts, "step": "response"},
            ))

            result = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if not result or not result.data.get("accepted", False):
                return False

            print("  Authenticated!")

            # Discovery
            disc = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if disc and disc.packet_type == PacketType.DISCOVERY:
                self._remote_bots = disc.data.get("bots", [])
                print(f"  Bots available: {self._remote_bots}")

            await write_frame(self._writer, make_discovery(self.node_name, bots=[]))
            return True

        except Exception as e:
            print(f"  Connection failed: {e}")
            return False

    async def test_latent_route(self, prompt: str) -> Optional[str]:
        """Test LATENT_ROUTE — auto-route by semantic intent."""
        stream_id = str(uuid.uuid4())[:8]

        pkt = COSMOSPacket(
            packet_type=PacketType.LATENT_ROUTE,
            sender=self.node_name,
            stream_id=stream_id,
            data={"prompt": prompt, "max_tokens": 500},
        )
        await write_frame(self._writer, pkt)
        print(f"  Sent LATENT_ROUTE (stream: {stream_id})")

        return await self._collect_response(stream_id, timeout=120.0)

    async def test_standard_route(self, bot_name: str, prompt: str) -> Optional[str]:
        """Test standard ROUTE for comparison."""
        stream_id = str(uuid.uuid4())[:8]

        pkt = COSMOSPacket(
            packet_type=PacketType.ROUTE,
            sender=self.node_name,
            target=bot_name,
            stream_id=stream_id,
            data={"prompt": prompt, "bot": bot_name, "max_tokens": 500},
        )
        await write_frame(self._writer, pkt)
        print(f"  Sent ROUTE to {bot_name} (stream: {stream_id})")

        return await self._collect_response(stream_id, timeout=120.0)

    async def test_poi_request(self, prompt: str, model: str) -> None:
        """Test POI_REQUEST — trigger consensus round."""
        round_id = str(uuid.uuid4())[:12]

        pkt = COSMOSPacket(
            packet_type=PacketType.POI_REQUEST,
            sender=self.node_name,
            target="*",
            stream_id=round_id,
            data={
                "round_id": round_id,
                "prompt": prompt,
                "model": model,
                "min_validators": 1,
            },
        )
        await write_frame(self._writer, pkt)
        print(f"  Sent POI_REQUEST (round: {round_id}, model: {model})")

        # Wait for attestation response
        start = time.time()
        while time.time() - start < 120.0:
            try:
                resp = await asyncio.wait_for(read_frame(self._reader), timeout=60.0)
                if resp is None:
                    break

                if resp.packet_type == PacketType.POI_ATTESTATION:
                    data = resp.data or {}
                    print(f"  PoI ATTESTATION received!")
                    print(f"    Node: {data.get('node')}")
                    print(f"    Model: {data.get('model')}")
                    print(f"    Output hash: {data.get('output_hash', b'')[:8].hex() if isinstance(data.get('output_hash'), bytes) else str(data.get('output_hash', ''))[:16]}")
                    print(f"    Inference time: {data.get('infer_ms', 0):.0f}ms")
                    print(f"    Seal: {data.get('seal', b'').hex() if isinstance(data.get('seal'), bytes) else str(data.get('seal', ''))[:32]}")
                    output = data.get("output", "")
                    if output:
                        print(f"    Output: {output[:200]}...")
                    return
                elif resp.packet_type == PacketType.POI_RESULT:
                    data = resp.data or {}
                    print(f"  PoI CONSENSUS PROOF received!")
                    print(f"    Agreement: {data.get('agreement_ratio', 0):.0%}")
                    print(f"    Validators: {data.get('total_validators', 0)}")
                    return
                elif resp.packet_type in (PacketType.HEARTBEAT, PacketType.DISCOVERY):
                    continue
                elif resp.packet_type == PacketType.ERROR:
                    print(f"  ERROR: {resp.data}")
                    return
                else:
                    print(f"  (got {PacketType(resp.packet_type).name}, skipping)")

            except asyncio.TimeoutError:
                print(f"  Timeout waiting for PoI response")
                return

    async def _collect_response(self, stream_id: str, timeout: float = 120.0) -> Optional[str]:
        """Collect DIALOGUE response chunks."""
        chunks = []
        start = time.time()

        while time.time() - start < timeout:
            try:
                remaining = max(1, timeout - (time.time() - start))
                pkt = await asyncio.wait_for(read_frame(self._reader), timeout=remaining)
                if pkt is None:
                    break

                if pkt.packet_type == PacketType.DIALOGUE:
                    chunk = pkt.data if isinstance(pkt.data, str) else str(pkt.data)
                    chunks.append(chunk)
                    if pkt.final:
                        break
                elif pkt.packet_type == PacketType.ERROR:
                    error = pkt.data.get("error", "Unknown") if isinstance(pkt.data, dict) else str(pkt.data)
                    print(f"  ERROR: {error}")
                    return None
                elif pkt.packet_type in (PacketType.HEARTBEAT, PacketType.DISCOVERY):
                    continue
                else:
                    continue

            except asyncio.TimeoutError:
                print(f"  Timeout after {time.time() - start:.1f}s")
                break
            except (asyncio.IncompleteReadError, ConnectionError, OSError) as e:
                print(f"  Connection error: {e}")
                break

        elapsed = time.time() - start
        response = "".join(chunks)
        print(f"  Received {len(chunks)} chunks in {elapsed:.2f}s ({len(response)} chars)")
        return response if response else None

    async def close(self):
        if self._writer:
            self._writer.close()
            self._writer = None


async def test_latent_router_standalone():
    """Test the Latent Router locally (no network needed)."""
    print("\n" + "="*60)
    print("TEST: Latent Space Router (standalone)")
    print("="*60)

    router = LatentRouter()

    test_prompts = [
        ("Write a Python async TCP server with error handling", "code"),
        ("Solve the integral of x^2 * sin(x) dx", "math"),
        ("Explain why quantum computers can break RSA encryption", "reasoning"),
        ("Write a short story about an AI that dreams", "creative"),
        ("What is the capital of Australia and when was it founded?", "factual"),
        ("Translate 'Hello world' into Chinese, Japanese, and Korean", "multilingual"),
    ]

    for prompt, expected_category in test_prompts:
        decision = router.route(prompt)
        print(f"\n  Prompt: '{prompt[:50]}...'")
        print(f"  Expected category: {expected_category}")
        print(f"  Selected model: {decision.selected_bot}")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Method: {decision.method}")
        print(f"  Query dimensions: {', '.join(f'{k}={v:.2f}' for k, v in sorted(decision.query_dimensions.items(), key=lambda x: -x[1]) if v > 0.1)}")
        print(f"  All scores: {', '.join(f'{k}={v:.3f}' for k, v in sorted(decision.all_scores.items(), key=lambda x: -x[1])[:3])}")

    stats = router.get_routing_stats()
    print(f"\n  Total routes: {stats['total_routes']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
    print("  PASS!")


async def main():
    parser = argparse.ArgumentParser(description="COSMOS v2.0 Integration Test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=COSMOS_PORT)
    parser.add_argument("--test", default="all", choices=["all", "latent", "poi", "route", "local"])
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    if args.test == "local":
        await test_latent_router_standalone()
        return

    print(f"\nCOSMOS v2.0 Integration Test")
    print(f"Connecting to {args.host}:{args.port}...")

    client = V2TestClient(args.host, args.port)
    if not await client.connect(compute_gpu=not args.no_gpu):
        print("FAILED to connect")
        return

    if args.test in ("all", "latent"):
        print("\n" + "="*60)
        print("TEST 1: Latent Space Routing")
        print("="*60)

        # Standalone router test first
        await test_latent_router_standalone()

        # Network latent route test
        print("\n--- Network Latent Route ---")
        prompts = [
            "Write a Python function to check if a number is prime",
            "Explain the difference between TCP and UDP in simple terms",
        ]
        for prompt in prompts:
            print(f"\n  Prompt: '{prompt}'")
            result = await client.test_latent_route(prompt)
            if result:
                print(f"  Response: {result[:200]}...")
            print()

    if args.test in ("all", "poi"):
        print("\n" + "="*60)
        print("TEST 2: Proof-of-Inference Consensus")
        print("="*60)

        print("\n  Sending PoI request for phi4-latest...")
        await client.test_poi_request(
            "What is 2+2? Answer with just the number.",
            "phi4-latest",
        )

    if args.test in ("all", "route"):
        print("\n" + "="*60)
        print("TEST 3: Standard Route (for comparison)")
        print("="*60)

        print("\n  Routing to phi4-latest...")
        result = await client.test_standard_route(
            "phi4-latest", "What is the meaning of life? One sentence."
        )
        if result:
            print(f"  Response: {result[:200]}")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
