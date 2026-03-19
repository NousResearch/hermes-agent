"""
COSMOS Test Client — Standalone process that connects to a running COSMOS node.

Bypasses the singleton limitation by speaking raw COSMOS protocol over TCP.

Usage:
    python -m cosmos.network.cosmos_test_client --bot qwen3-coder-next-latest --prompt "Write a prime checker"
    python -m cosmos.network.cosmos_test_client --status
"""
import asyncio
import argparse
import time
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
from . import COSMOS_PORT


class COSMOSTestClient:
    """
    Standalone COSMOS client that connects as a lightweight test node.
    Works from any process — no singleton needed.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = COSMOS_PORT,
                 node_name: str = "test-client"):
        self.host = host
        self.port = port
        self.node_name = node_name
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._seed = get_or_create_seed()
        self._gpu_fp = b""
        self._identity = b""
        self._remote_bots = []

    async def connect(self, compute_gpu: bool = True) -> bool:
        """Connect to the local COSMOS node and authenticate."""
        # Compute GPU fingerprint (or use empty for test)
        if compute_gpu:
            loop = asyncio.get_event_loop()
            self._gpu_fp = await loop.run_in_executor(None, compute_gpu_fingerprint)
        self._identity = derive_node_identity(self._seed, self.node_name, self._gpu_fp)

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=10.0,
            )

            # TCP_NODELAY
            sock = self._writer.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Send HELLO
            await write_frame(self._writer, make_hello(
                self.node_name,
                self._gpu_fp,
                {"version": 1, "bots": []},
            ))

            # Receive challenge
            resp = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if not resp or resp.packet_type != PacketType.VERIFY:
                print(f"Expected VERIFY challenge, got {resp.packet_type if resp else 'nothing'}")
                return False

            challenge = resp.data.get("challenge", b"")
            ts = time.time()

            # Solve challenge
            response = solve_challenge(self._identity, challenge, ts)
            await write_frame(self._writer, COSMOSPacket(
                packet_type=PacketType.VERIFY,
                sender=self.node_name,
                target=resp.sender,
                data={"response": response, "timestamp": ts, "step": "response"},
            ))

            # Receive verification result
            result = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if not result or result.packet_type != PacketType.VERIFY:
                print(f"Expected VERIFY result, got {result.packet_type if result else 'nothing'}")
                return False

            if not result.data.get("accepted", False):
                print(f"Auth rejected: {result.data.get('reason', 'unknown')}")
                return False

            print(f"Authenticated with COSMOS node!")

            # Receive DISCOVERY (server sends its bot list)
            disc = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if disc and disc.packet_type == PacketType.DISCOVERY:
                self._remote_bots = disc.data.get("bots", [])
                print(f"Remote bots available: {self._remote_bots}")

            # Send our own DISCOVERY (empty — we're a test client)
            await write_frame(self._writer, make_discovery(self.node_name, bots=[]))

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def query(self, bot_name: str, prompt: str,
                    max_tokens: int = 4000, timeout: float = 300.0) -> Optional[str]:
        """Send a ROUTE request and collect the DIALOGUE response."""
        if not self._writer:
            print("Not connected")
            return None

        stream_id = str(uuid.uuid4())[:8]

        # Send ROUTE request
        route_pkt = COSMOSPacket(
            packet_type=PacketType.ROUTE,
            sender=self.node_name,
            target=bot_name,
            stream_id=stream_id,
            data={"prompt": prompt, "bot": bot_name, "max_tokens": max_tokens},
        )
        await write_frame(self._writer, route_pkt)
        print(f"Sent ROUTE request for '{bot_name}' (stream: {stream_id})")

        # Collect response chunks — large models (80B+) can take 2+ minutes
        chunks = []
        start = time.time()

        while time.time() - start < timeout:
            try:
                remaining = max(1, timeout - (time.time() - start))
                pkt = await asyncio.wait_for(
                    read_frame(self._reader),
                    timeout=remaining,
                )
                if pkt is None:
                    break

                if pkt.packet_type == PacketType.DIALOGUE:
                    chunk = pkt.data if isinstance(pkt.data, str) else str(pkt.data)
                    chunks.append(chunk)
                    if pkt.final:
                        break
                elif pkt.packet_type == PacketType.ERROR:
                    error = pkt.data.get("error", "Unknown") if isinstance(pkt.data, dict) else str(pkt.data)
                    print(f"ERROR from node: {error}")
                    return None
                elif pkt.packet_type == PacketType.HEARTBEAT:
                    continue  # Skip heartbeats
                elif pkt.packet_type == PacketType.DISCOVERY:
                    continue  # Skip discovery updates
                else:
                    print(f"  (got {pkt.packet_type}, skipping)")

            except asyncio.TimeoutError:
                print(f"Timeout waiting for response after {time.time() - start:.1f}s")
                break
            except (asyncio.IncompleteReadError, ConnectionError, OSError) as e:
                print(f"Connection error: {e}")
                break

        elapsed = time.time() - start
        response = "".join(chunks)
        print(f"Received {len(chunks)} chunks in {elapsed:.2f}s ({len(response)} chars)")
        return response if response else None

    async def get_status(self) -> dict:
        """Get all bots visible through this node."""
        return {
            "connected": self._writer is not None,
            "remote_bots": self._remote_bots,
        }

    async def close(self):
        """Close the connection."""
        if self._writer:
            self._writer.close()
            self._writer = None


async def main():
    parser = argparse.ArgumentParser(description="COSMOS Test Client")
    parser.add_argument("--host", default="127.0.0.1", help="COSMOS node host")
    parser.add_argument("--port", type=int, default=COSMOS_PORT, help="COSMOS node port")
    parser.add_argument("--bot", help="Bot to query")
    parser.add_argument("--prompt", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--status", action="store_true", help="Just show status")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU fingerprint")
    args = parser.parse_args()

    client = COSMOSTestClient(args.host, args.port)

    print(f"Connecting to COSMOS node at {args.host}:{args.port}...")
    if not await client.connect(compute_gpu=not args.no_gpu):
        print("FAILED to connect/authenticate")
        return

    if args.status:
        status = await client.get_status()
        print(f"\nStatus: {status}")
    elif args.bot and args.prompt:
        print(f"\nQuerying bot '{args.bot}'...")
        response = await client.query(args.bot, args.prompt, args.max_tokens)
        if response:
            print(f"\n{'='*60}")
            print(response)
            print(f"{'='*60}")
        else:
            print("\nNo response received.")
    else:
        print("\nAvailable bots:", client._remote_bots)
        print("Use --bot NAME --prompt TEXT to query a bot")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
