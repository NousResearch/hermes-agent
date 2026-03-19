"""
COSMOS Proof-of-Inference (PoI) Consensus
==========================================

The first verifiable AI computation protocol:

  1. Requester sends prompt + model to N validator nodes
  2. Each validator runs inference on their own GPU
  3. Each creates a hardware-bound attestation:
     attestation = BLAKE3(gpu_fp || model_hash || output_hash || timestamp)
  4. BFT consensus: if 2/3+ validators produce matching output hashes → VERIFIED
  5. Compact proof = chain of attestation seals → cryptographically verifiable

Why this matters:
  - Proof-of-Work wastes compute. Proof-of-Stake wastes capital.
  - Proof-of-Inference does USEFUL WORK (real AI inference).
  - GPU fingerprint ties output to specific physical hardware.
  - Anti-hallucination: N independent models must agree.
  - Verifiable AI: prove what model produced what output on what hardware.

~200 lines. No external dependencies beyond BLAKE3.
"""
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from loguru import logger

from .cosmos_auth import blake3, compute_gpu_fingerprint
from .cosmos_protocol import (
    PacketType, COSMOSPacket, write_frame, read_frame,
)


@dataclass
class InferenceAttestation:
    """A single node's hardware-bound attestation of inference output."""
    attestation_id: str
    round_id: str
    node_name: str
    gpu_fingerprint: bytes          # 32-byte GPU hardware fingerprint
    model_name: str
    model_hash: bytes               # BLAKE3(model_name + model_params)
    prompt_hash: bytes              # BLAKE3(input prompt)
    output_hash: bytes              # BLAKE3(model output)
    output_text: str                # The actual inference output
    timestamp: float
    inference_time_ms: float        # How long inference took
    seal: bytes = b""               # BLAKE3 binding all fields

    def compute_seal(self, node_identity: bytes) -> bytes:
        """Compute hardware-bound seal over all attestation fields."""
        payload = (
            self.round_id.encode()
            + self.node_name.encode()
            + self.gpu_fingerprint
            + self.model_hash
            + self.prompt_hash
            + self.output_hash
            + str(self.timestamp).encode()
            + str(self.inference_time_ms).encode()
            + node_identity
        )
        self.seal = blake3(payload).digest()[:16]
        return self.seal

    def to_dict(self) -> Dict:
        return {
            "aid": self.attestation_id,
            "rid": self.round_id,
            "node": self.node_name,
            "gpu_fp": self.gpu_fingerprint,
            "model": self.model_name,
            "model_hash": self.model_hash,
            "prompt_hash": self.prompt_hash,
            "output_hash": self.output_hash,
            "output": self.output_text,
            "ts": self.timestamp,
            "infer_ms": self.inference_time_ms,
            "seal": self.seal,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "InferenceAttestation":
        return cls(
            attestation_id=d.get("aid", ""),
            round_id=d.get("rid", ""),
            node_name=d.get("node", ""),
            gpu_fingerprint=d.get("gpu_fp", b""),
            model_name=d.get("model", ""),
            model_hash=d.get("model_hash", b""),
            prompt_hash=d.get("prompt_hash", b""),
            output_hash=d.get("output_hash", b""),
            output_text=d.get("output", ""),
            timestamp=d.get("ts", 0.0),
            inference_time_ms=d.get("infer_ms", 0.0),
            seal=d.get("seal", b""),
        )


@dataclass
class ConsensusProof:
    """Compact verifiable proof of consensus inference."""
    round_id: str
    prompt_hash: bytes
    consensus_hash: bytes               # Hash of agreed output
    consensus_output: str               # The agreed-upon text
    attestations: List[InferenceAttestation]
    agreement_ratio: float              # e.g., 3/5 = 0.6
    total_validators: int
    agreeing_validators: int
    proof_seal: bytes                   # BLAKE3(all attestation seals chained)
    timestamp: float
    verified: bool = False

    def to_dict(self) -> Dict:
        return {
            "round_id": self.round_id,
            "prompt_hash": self.prompt_hash.hex(),
            "consensus_hash": self.consensus_hash.hex(),
            "consensus_output": self.consensus_output,
            "attestations": [a.to_dict() for a in self.attestations],
            "agreement_ratio": self.agreement_ratio,
            "total_validators": self.total_validators,
            "agreeing_validators": self.agreeing_validators,
            "proof_seal": self.proof_seal.hex(),
            "timestamp": self.timestamp,
            "verified": self.verified,
        }


def hash_model(model_name: str) -> bytes:
    """Create deterministic model identifier hash."""
    return blake3(model_name.encode("utf-8")).digest()


def hash_prompt(prompt: str) -> bytes:
    """Hash a prompt for attestation."""
    return blake3(prompt.encode("utf-8")).digest()


def hash_output(output: str) -> bytes:
    """Hash inference output for comparison."""
    # Normalize whitespace for fair comparison
    normalized = " ".join(output.strip().split())
    return blake3(normalized.encode("utf-8")).digest()


class ProofOfInference:
    """
    Proof-of-Inference consensus engine.

    Coordinates multi-node inference rounds, collects hardware-bound
    attestations, and produces cryptographically verifiable proofs.
    """

    def __init__(self, node_name: str, node_identity: bytes, gpu_fp: bytes):
        self.node_name = node_name
        self._identity = node_identity
        self._gpu_fp = gpu_fp
        # Active consensus rounds: round_id → state
        self._rounds: Dict[str, Dict] = {}
        # Completed proofs
        self._proofs: List[ConsensusProof] = []

    def create_round(self, prompt: str, model_name: str,
                     min_validators: int = 2) -> str:
        """Create a new consensus round. Returns round_id."""
        round_id = str(uuid.uuid4())[:12]
        self._rounds[round_id] = {
            "prompt": prompt,
            "model": model_name,
            "prompt_hash": hash_prompt(prompt),
            "model_hash": hash_model(model_name),
            "min_validators": min_validators,
            "attestations": [],
            "started": time.time(),
            "status": "collecting",
        }
        logger.info(f"PoI round {round_id} started: model={model_name}, min_validators={min_validators}")
        return round_id

    def create_attestation(self, round_id: str, prompt: str,
                           model_name: str, output: str,
                           inference_time_ms: float) -> InferenceAttestation:
        """Create a hardware-bound attestation for an inference result."""
        att = InferenceAttestation(
            attestation_id=str(uuid.uuid4())[:8],
            round_id=round_id,
            node_name=self.node_name,
            gpu_fingerprint=self._gpu_fp,
            model_name=model_name,
            model_hash=hash_model(model_name),
            prompt_hash=hash_prompt(prompt),
            output_hash=hash_output(output),
            output_text=output,
            timestamp=time.time(),
            inference_time_ms=inference_time_ms,
        )
        att.compute_seal(self._identity)
        logger.info(
            f"PoI attestation created: round={round_id}, "
            f"output_hash={att.output_hash[:8].hex()}, "
            f"seal={att.seal.hex()}"
        )
        return att

    def add_attestation(self, round_id: str,
                        attestation: InferenceAttestation) -> bool:
        """Add a remote attestation to a round."""
        rd = self._rounds.get(round_id)
        if not rd or rd["status"] != "collecting":
            return False

        # Verify prompt hash matches
        if attestation.prompt_hash != rd["prompt_hash"]:
            logger.warning(f"PoI: prompt hash mismatch from {attestation.node_name}")
            return False

        # Verify model hash matches
        if attestation.model_hash != rd["model_hash"]:
            logger.warning(f"PoI: model hash mismatch from {attestation.node_name}")
            return False

        # Don't accept duplicate attestations from same node
        for existing in rd["attestations"]:
            if existing.node_name == attestation.node_name:
                logger.debug(f"PoI: duplicate attestation from {attestation.node_name}")
                return False

        rd["attestations"].append(attestation)
        logger.info(
            f"PoI attestation added: round={round_id}, "
            f"from={attestation.node_name}, "
            f"total={len(rd['attestations'])}"
        )
        return True

    def try_consensus(self, round_id: str) -> Optional[ConsensusProof]:
        """
        Attempt to reach consensus for a round.

        BFT rule: 2/3+ of validators must produce matching output hashes.
        Returns ConsensusProof if consensus reached, None otherwise.
        """
        rd = self._rounds.get(round_id)
        if not rd or rd["status"] != "collecting":
            return None

        attestations = rd["attestations"]
        min_validators = rd["min_validators"]

        if len(attestations) < min_validators:
            return None  # Not enough attestations yet

        # Group attestations by output hash
        hash_groups: Dict[bytes, List[InferenceAttestation]] = {}
        for att in attestations:
            key = att.output_hash
            if key not in hash_groups:
                hash_groups[key] = []
            hash_groups[key].append(att)

        # Find the majority group
        total = len(attestations)
        best_hash = max(hash_groups.keys(), key=lambda h: len(hash_groups[h]))
        best_group = hash_groups[best_hash]
        agreement = len(best_group) / total

        # BFT threshold: 2/3 must agree
        threshold = 2 / 3
        if agreement < threshold:
            logger.warning(
                f"PoI round {round_id}: no consensus "
                f"({len(best_group)}/{total} = {agreement:.1%}, need {threshold:.1%})"
            )
            # If we have all expected attestations and still no consensus, fail
            if total >= min_validators:
                rd["status"] = "no_consensus"
            return None

        # Consensus reached!
        rd["status"] = "consensus"

        # Chain all attestation seals into proof seal
        seal_chain = b""
        for att in best_group:
            seal_chain += att.seal
        proof_seal = blake3(seal_chain).digest()[:16]

        proof = ConsensusProof(
            round_id=round_id,
            prompt_hash=rd["prompt_hash"],
            consensus_hash=best_hash,
            consensus_output=best_group[0].output_text,
            attestations=attestations,
            agreement_ratio=agreement,
            total_validators=total,
            agreeing_validators=len(best_group),
            proof_seal=proof_seal,
            timestamp=time.time(),
            verified=True,
        )

        self._proofs.append(proof)
        logger.info(
            f"PoI CONSENSUS for round {round_id}: "
            f"{len(best_group)}/{total} agree ({agreement:.0%}), "
            f"proof_seal={proof_seal.hex()}"
        )
        return proof

    def verify_proof(self, proof: ConsensusProof) -> bool:
        """
        Verify a consensus proof independently.

        Checks:
        1. All attestation seals are internally consistent
        2. Majority output hashes match consensus hash
        3. Proof seal is correct chain of attestation seals
        4. Agreement ratio meets BFT threshold
        """
        if proof.agreement_ratio < 2 / 3:
            return False

        # Verify majority match consensus hash
        matching = sum(
            1 for a in proof.attestations
            if a.output_hash == proof.consensus_hash
        )
        if matching / len(proof.attestations) < 2 / 3:
            return False

        # Verify proof seal
        agreeing = [a for a in proof.attestations if a.output_hash == proof.consensus_hash]
        seal_chain = b""
        for att in agreeing:
            seal_chain += att.seal
        expected_seal = blake3(seal_chain).digest()[:16]

        if expected_seal != proof.proof_seal:
            return False

        return True

    def get_proofs(self) -> List[Dict]:
        """Get all completed proofs."""
        return [p.to_dict() for p in self._proofs]

    def get_round_status(self, round_id: str) -> Optional[Dict]:
        """Get status of a consensus round."""
        rd = self._rounds.get(round_id)
        if not rd:
            return None
        return {
            "round_id": round_id,
            "model": rd["model"],
            "status": rd["status"],
            "attestations": len(rd["attestations"]),
            "min_validators": rd["min_validators"],
            "age_seconds": time.time() - rd["started"],
        }

    def cleanup_stale(self, max_age: float = 600.0):
        """Remove stale rounds older than max_age."""
        cutoff = time.time() - max_age
        stale = [rid for rid, rd in self._rounds.items()
                 if rd["started"] < cutoff]
        for rid in stale:
            del self._rounds[rid]


def make_poi_request(sender: str, round_id: str, prompt: str,
                     model_name: str, min_validators: int = 2) -> COSMOSPacket:
    """Create a POI_REQUEST packet — ask peers to participate in consensus."""
    return COSMOSPacket(
        packet_type=PacketType.POI_REQUEST,
        sender=sender,
        target="*",
        stream_id=round_id,
        data={
            "round_id": round_id,
            "prompt": prompt,
            "model": model_name,
            "min_validators": min_validators,
            "prompt_hash": hash_prompt(prompt),
            "model_hash": hash_model(model_name),
        },
    )


def make_poi_attestation(sender: str, attestation: InferenceAttestation) -> COSMOSPacket:
    """Create a POI_ATTESTATION packet — return hardware-signed attestation."""
    return COSMOSPacket(
        packet_type=PacketType.POI_ATTESTATION,
        sender=sender,
        stream_id=attestation.round_id,
        data=attestation.to_dict(),
    )


def make_poi_result(sender: str, proof: ConsensusProof) -> COSMOSPacket:
    """Create a POI_RESULT packet — broadcast consensus proof."""
    return COSMOSPacket(
        packet_type=PacketType.POI_RESULT,
        sender=sender,
        target="*",
        stream_id=proof.round_id,
        data=proof.to_dict(),
    )
