"""
Cosmos Federated Population Manager - Distributed Evolution

Integrates genetic_optimizer.py with p2p.py for planetary-scale
distributed evolution across Cosmos instances.

AGI Cohesion Features:
- Island model evolution with P2P migration
- Privacy-preserving fitness aggregation
- Federated genome averaging
- Distributed selection pressure
- Cross-collective evolution

AGI v1.8 Quantum Enhancements:
- Quantum population generation using superposition
- Quantum self-improvement loops via IBM Quantum
- Simulator-first strategy (unlimited) with hardware validation (10min/month)

This enables AGI collectives to evolve together while maintaining
privacy and allowing diverse evolutionary paths.
"""

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from collections import defaultdict

from loguru import logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


@dataclass
class FederatedEvolutionConfig:
    """Configuration for federated population evolution."""
    # Local evolution
    local_population_size: int = 20
    local_generations_per_sync: int = 5
    elite_ratio: float = 0.2
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7

    # Federation
    migration_rate: float = 0.1  # Fraction of population to migrate
    migration_interval_generations: int = 5
    min_peers_for_federation: int = 2
    max_immigrants_per_sync: int = 5

    # Privacy
    privacy_epsilon: float = 1.0
    anonymize_fitness: bool = True
    share_top_k_only: int = 5  # Only share top K genomes

    # Aggregation
    federated_averaging: bool = True
    fitness_aggregation: str = "weighted_mean"  # "mean", "weighted_mean", "max"


@dataclass
class MigrantGenome:
    """A genome migrating from another node."""
    genome_hash: str
    genes: Dict[str, float]
    fitness_score: float
    source_node: str
    generation: int
    received_at: datetime = field(default_factory=datetime.now)


@dataclass
class FederatedFitnessReport:
    """Aggregated fitness from the federation."""
    genome_hash: str
    local_fitness: Dict[str, float]
    federated_fitness: Dict[str, float]
    peer_count: int
    confidence: float  # Higher = more peers agree


class FederatedPopulationManager:
    """
    Manages distributed population evolution across P2P network.

    Implements island model with migration:
    1. Each node evolves locally for N generations
    2. Top performers migrate to random peers
    3. Incoming migrants compete with local population
    4. Fitness is aggregated across the federation

    Features:
    - Privacy-preserving genome sharing
    - Federated fitness aggregation
    - Cross-node genome migration
    - Distributed selection pressure
    """

    def __init__(
        self,
        config: Optional[FederatedEvolutionConfig] = None,
        genetic_optimizer=None,
        swarm_fabric=None,
        fitness_tracker=None,
    ):
        self.config = config or FederatedEvolutionConfig()
        self.optimizer = genetic_optimizer
        self.swarm = swarm_fabric
        self.fitness_tracker = fitness_tracker

        # Local state
        self.node_id: str = ""
        self.generation: int = 0
        self.local_best_fitness: float = 0.0

        # Migration tracking
        self.incoming_migrants: List[MigrantGenome] = []
        self.outgoing_migrations: int = 0
        self.successful_integrations: int = 0

        # Federated fitness aggregation
        self.federated_fitness: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )  # genome_hash -> metric -> [scores from peers]
        self.peer_fitness_reports: Dict[str, List[Dict]] = defaultdict(list)

        # Evolution history
        self.federation_history: List[Dict] = []

        # Callbacks
        self._on_migration_received: List[Callable] = []
        self._on_fitness_aggregated: List[Callable] = []

        self._lock = asyncio.Lock()
        self._is_running = False

        # AGI v1.8: Quantum evolution integration
        self._quantum_available = False
        self._quantum_optimizer = None
        try:
            from Cosmos.integration.quantum import QISKIT_AVAILABLE, get_quantum_provider
            from Cosmos.integration.quantum.ibm_quantum import QuantumGeneticOptimizer
            self._quantum_available = QISKIT_AVAILABLE
            if QISKIT_AVAILABLE:
                provider = get_quantum_provider()
                if provider:
                    self._quantum_optimizer = QuantumGeneticOptimizer(provider, num_qubits=8)
                    logger.info("Quantum self-improvement enabled (IBM Quantum)")
        except ImportError:
            pass

    def set_node_id(self, node_id: str):
        """Set the local node ID."""
        self.node_id = node_id

    def connect_swarm(self, swarm_fabric):
        """Connect to P2P swarm fabric."""
        self.swarm = swarm_fabric
        if hasattr(swarm_fabric, 'node_id'):
            self.node_id = swarm_fabric.node_id

    def connect_optimizer(self, genetic_optimizer):
        """Connect to local genetic optimizer."""
        self.optimizer = genetic_optimizer

    async def start(self):
        """Start federated evolution loop."""
        if self._is_running:
            return

        self._is_running = True
        logger.info(f"Federated Population Manager started (node={self.node_id})")

        # Start background evolution loop
        asyncio.create_task(self._evolution_loop())

    async def stop(self):
        """Stop federated evolution."""
        self._is_running = False
        logger.info("Federated Population Manager stopped")

    async def _evolution_loop(self):
        """Main federated evolution loop."""
        while self._is_running:
            try:
                # 1. Local evolution for N generations
                for _ in range(self.config.local_generations_per_sync):
                    if self.optimizer:
                        await self.optimizer.evolve_generation()
                        self.generation = self.optimizer.generation

                # 2. Process incoming migrants
                await self._process_migrants()

                # 3. Aggregate federated fitness
                await self._aggregate_federated_fitness()

                # 4. Migrate top performers to peers
                if self.generation % self.config.migration_interval_generations == 0:
                    await self._migrate_top_performers()

                # 5. Broadcast fitness reports
                await self._broadcast_fitness()

                # Record history
                self._record_federation_event()

                # Small delay between sync cycles
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Federation evolution error: {e}")
                await asyncio.sleep(5.0)

    async def _process_migrants(self):
        """Process incoming migrant genomes."""
        async with self._lock:
            if not self.incoming_migrants or not self.optimizer:
                return

            # Sort migrants by fitness
            migrants = sorted(
                self.incoming_migrants,
                key=lambda m: m.fitness_score,
                reverse=True,
            )

            # Take top migrants up to limit
            accepted = 0
            for migrant in migrants[:self.config.max_immigrants_per_sync]:
                # Check if migrant is better than worst local genome
                if self.optimizer.population:
                    worst_local = min(
                        self.optimizer.population,
                        key=lambda g: g.total_fitness()
                    )
                    if migrant.fitness_score > worst_local.total_fitness():
                        # Replace worst with migrant
                        await self._integrate_migrant(migrant, worst_local)
                        accepted += 1

            self.successful_integrations += accepted
            self.incoming_migrants = []

            if accepted > 0:
                logger.info(f"Integrated {accepted} migrant genomes")

    async def _integrate_migrant(self, migrant: MigrantGenome, replace_genome):
        """Integrate a migrant genome into local population."""
        if not self.optimizer:
            return

        from Cosmos.evolution.genetic_optimizer import Genome, Gene

        # Create new Genome from migrant data
        genes = {}
        for name, value in migrant.genes.items():
            if name in self.optimizer.gene_definitions:
                defn = self.optimizer.gene_definitions[name]
                genes[name] = Gene(
                    name=name,
                    value=value,
                    min_val=defn["min"],
                    max_val=defn["max"],
                    mutation_sigma=defn["sigma"],
                )

        if not genes:
            return

        new_genome = Genome(
            id=f"migrant_{migrant.genome_hash[:8]}_{self.generation}",
            genes=genes,
            generation=self.generation,
            parent_ids=[f"remote:{migrant.source_node}"],
        )

        # Replace in population
        try:
            idx = self.optimizer.population.index(replace_genome)
            self.optimizer.population[idx] = new_genome
        except ValueError:
            # Genome not found, append instead
            self.optimizer.population.append(new_genome)

    async def _migrate_top_performers(self):
        """Migrate top genomes to random peers."""
        if not self.swarm or not self.optimizer:
            return

        if not self.optimizer.population:
            return

        # Get peer list
        peer_ids = list(self.swarm.peers.keys())
        if len(peer_ids) < self.config.min_peers_for_federation:
            return

        # Select top genomes to migrate
        sorted_pop = sorted(
            self.optimizer.population,
            key=lambda g: g.total_fitness(),
            reverse=True,
        )

        migrate_count = min(
            self.config.share_top_k_only,
            int(len(sorted_pop) * self.config.migration_rate),
        )

        for genome in sorted_pop[:migrate_count]:
            # Select random peer
            target_peer = random.choice(peer_ids)

            # Prepare genome data (with optional anonymization)
            genome_data = self._prepare_genome_for_migration(genome)

            # Send via P2P
            await self.swarm.migrate_genome(
                target_peer_id=target_peer,
                genome_data=genome_data,
                fitness_score=genome.total_fitness(),
                generation=self.generation,
            )

            self.outgoing_migrations += 1

        logger.debug(f"Migrated {migrate_count} genomes to peers")

    def _prepare_genome_for_migration(self, genome) -> Dict:
        """Prepare genome for migration (with privacy options)."""
        genes_dict = {name: gene.value for name, gene in genome.genes.items()}

        if self.config.anonymize_fitness:
            # Add small noise to gene values
            import random
            genes_dict = {
                name: value + random.gauss(0, 0.01)
                for name, value in genes_dict.items()
            }

        return {
            "id": genome.id,
            "genes": genes_dict,
            "fitness_scores": genome.fitness_scores,
            "generation": genome.generation,
        }

    async def _broadcast_fitness(self):
        """Broadcast fitness reports to federation."""
        if not self.swarm or not self.optimizer:
            return

        if not self.optimizer.population:
            return

        # Share top genomes' fitness
        sorted_pop = sorted(
            self.optimizer.population,
            key=lambda g: g.total_fitness(),
            reverse=True,
        )

        for genome in sorted_pop[:self.config.share_top_k_only]:
            if not genome.fitness_scores:
                continue

            # Anonymize genome identifier
            genome_hash = hashlib.sha256(
                f"{genome.id}:{self.generation}".encode()
            ).hexdigest()[:16]

            # Optionally add noise to fitness
            fitness = genome.fitness_scores
            if self.config.anonymize_fitness:
                import random
                fitness = {
                    name: max(0, min(1, score + random.gauss(0, 0.01)))
                    for name, score in fitness.items()
                }

            await self.swarm.broadcast_fitness(
                genome_hash=genome_hash,
                fitness_scores=fitness,
                generation=self.generation,
            )

    async def _aggregate_federated_fitness(self):
        """Aggregate fitness from peer reports."""
        async with self._lock:
            aggregated_reports = []

            for genome_hash, metrics in self.federated_fitness.items():
                if not metrics:
                    continue

                aggregated = {}
                peer_count = 0

                for metric_name, scores in metrics.items():
                    if not scores:
                        continue

                    peer_count = max(peer_count, len(scores))

                    if self.config.fitness_aggregation == "mean":
                        aggregated[metric_name] = sum(scores) / len(scores)
                    elif self.config.fitness_aggregation == "max":
                        aggregated[metric_name] = max(scores)
                    else:  # weighted_mean
                        # Weight by recency (assuming scores are in order)
                        weights = [1.0 + 0.1 * i for i in range(len(scores))]
                        weighted_sum = sum(s * w for s, w in zip(scores, weights))
                        aggregated[metric_name] = weighted_sum / sum(weights)

                if aggregated:
                    # Calculate confidence based on peer agreement
                    variance = sum(
                        sum((s - aggregated.get(m, 0)) ** 2 for s in scores)
                        for m, scores in metrics.items()
                    )
                    confidence = 1.0 / (1.0 + variance)

                    report = FederatedFitnessReport(
                        genome_hash=genome_hash,
                        local_fitness={},  # Would need to match with local genomes
                        federated_fitness=aggregated,
                        peer_count=peer_count,
                        confidence=confidence,
                    )
                    aggregated_reports.append(report)

                    # Notify callbacks
                    for callback in self._on_fitness_aggregated:
                        try:
                            await callback(report)
                        except Exception as e:
                            logger.debug(f"Fitness aggregation callback error: {e}")

            # Clear old federated fitness data
            self.federated_fitness.clear()

    def receive_migrant(
        self,
        genome_data: Dict,
        fitness_score: float,
        source_node: str,
        generation: int,
    ):
        """
        Receive a migrant genome from P2P network.

        Called by P2P message handler when GOSSIP_GENOME_MIGRATION received.
        """
        genome_hash = hashlib.sha256(
            json.dumps(genome_data.get("genes", {}), sort_keys=True).encode()
        ).hexdigest()[:16]

        migrant = MigrantGenome(
            genome_hash=genome_hash,
            genes=genome_data.get("genes", {}),
            fitness_score=fitness_score,
            source_node=source_node,
            generation=generation,
        )

        self.incoming_migrants.append(migrant)

        # Notify callbacks
        for callback in self._on_migration_received:
            try:
                asyncio.create_task(callback(migrant))
            except Exception as e:
                logger.debug(f"Migration callback error: {e}")

    def receive_fitness_report(
        self,
        genome_hash: str,
        fitness_scores: Dict[str, float],
        peer_id: str,
    ):
        """
        Receive fitness report from P2P network.

        Called by P2P message handler when GOSSIP_FITNESS received.
        """
        for metric_name, score in fitness_scores.items():
            self.federated_fitness[genome_hash][metric_name].append(score)

        # Keep bounded
        for genome_hash, metrics in self.federated_fitness.items():
            for metric_name, scores in metrics.items():
                if len(scores) > 20:
                    self.federated_fitness[genome_hash][metric_name] = scores[-10:]

    def _record_federation_event(self):
        """Record federation event in history."""
        event = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "local_population_size": len(self.optimizer.population) if self.optimizer else 0,
            "local_best_fitness": self.local_best_fitness,
            "outgoing_migrations": self.outgoing_migrations,
            "successful_integrations": self.successful_integrations,
            "pending_migrants": len(self.incoming_migrants),
            "federated_genomes_tracked": len(self.federated_fitness),
        }
        self.federation_history.append(event)

        # Keep bounded
        if len(self.federation_history) > 1000:
            self.federation_history = self.federation_history[-500:]

    def on_migration_received(self, callback: Callable):
        """Register callback for migration events."""
        self._on_migration_received.append(callback)

    def on_fitness_aggregated(self, callback: Callable):
        """Register callback for fitness aggregation events."""
        self._on_fitness_aggregated.append(callback)

    def get_stats(self) -> Dict:
        """Get federation statistics."""
        return {
            "node_id": self.node_id,
            "generation": self.generation,
            "is_running": self._is_running,
            "local_best_fitness": self.local_best_fitness,
            "outgoing_migrations": self.outgoing_migrations,
            "successful_integrations": self.successful_integrations,
            "pending_migrants": len(self.incoming_migrants),
            "federated_genomes_tracked": len(self.federated_fitness),
            "recent_history": self.federation_history[-5:],
            "config": {
                "local_population_size": self.config.local_population_size,
                "migration_rate": self.config.migration_rate,
                "privacy_epsilon": self.config.privacy_epsilon,
            },
        }

    async def quantum_self_improve_population(
        self,
        fitness_func: Optional[Callable] = None,
        grover_iterations: int = 1,
        use_hardware: bool = False,
    ) -> int:
        """
        AGI v1.8: Quantum self-improvement using superposition-based population diversity.

        Uses quantum genetic algorithm to generate diverse population variants,
        potentially finding optimal configurations faster than classical evolution.

        Args:
            fitness_func: Optional fitness function (uses optimizer's if None)
            grover_iterations: Number of Grover iterations for amplitude amplification
            use_hardware: Use IBM Quantum hardware (limited to 10min/month)

        Returns:
            Number of improved genomes integrated into population
        """
        if not self._quantum_available or not self._quantum_optimizer:
            logger.debug("Quantum not available, skipping quantum self-improvement")
            return 0

        if not self.optimizer or not self.optimizer.population:
            return 0

        try:
            # Get current best genome as baseline
            sorted_pop = sorted(
                self.optimizer.population,
                key=lambda g: g.total_fitness(),
                reverse=True,
            )
            best_genome = sorted_pop[0]
            baseline_fitness = best_genome.total_fitness()

            # Define fitness function for quantum optimization
            def quantum_fitness(bitstring: str) -> float:
                if fitness_func:
                    return fitness_func(bitstring)
                # Default: count 1s (maximize diversity)
                return sum(int(b) for b in bitstring) / len(bitstring)

            # Generate quantum population using superposition
            quantum_population = await self._quantum_optimizer.generate_quantum_population(
                population_size=min(10, len(self.optimizer.population)),
                fitness_func=quantum_fitness,
                grover_iterations=grover_iterations,
                prefer_hardware=use_hardware,
            )

            # Integrate improved quantum genomes into local population
            improved_count = 0
            from Cosmos.evolution.genetic_optimizer import Genome, Gene

            for bitstring, qfitness in quantum_population:
                # Convert bitstring to gene values
                if not bitstring or len(bitstring) < 8:
                    continue

                # Create genome from quantum sample
                genes = {}
                gene_names = list(self.optimizer.gene_definitions.keys())
                bits_per_gene = max(1, len(bitstring) // max(1, len(gene_names)))

                for i, name in enumerate(gene_names):
                    defn = self.optimizer.gene_definitions[name]
                    # Extract bits for this gene
                    start_bit = i * bits_per_gene
                    end_bit = min(start_bit + bits_per_gene, len(bitstring))
                    gene_bits = bitstring[start_bit:end_bit]

                    if gene_bits:
                        # Convert bits to normalized value
                        value_int = int(gene_bits, 2)
                        max_val = 2 ** len(gene_bits) - 1
                        normalized = value_int / max(1, max_val)
                        value = defn["min"] + normalized * (defn["max"] - defn["min"])

                        genes[name] = Gene(
                            name=name,
                            value=value,
                            min_val=defn["min"],
                            max_val=defn["max"],
                            mutation_sigma=defn["sigma"],
                        )

                if genes:
                    quantum_genome = Genome(
                        id=f"quantum_{self.generation}_{hash(bitstring) % 10000:04d}",
                        genes=genes,
                        generation=self.generation,
                        parent_ids=["quantum_self_improvement"],
                    )

                    # Check if quantum genome is better than worst local
                    worst_local = min(
                        self.optimizer.population,
                        key=lambda g: g.total_fitness()
                    )
                    if qfitness > worst_local.total_fitness():
                        # Replace worst with quantum genome
                        try:
                            idx = self.optimizer.population.index(worst_local)
                            self.optimizer.population[idx] = quantum_genome
                            improved_count += 1
                        except ValueError:
                            pass

            if improved_count > 0:
                logger.info(f"Quantum self-improvement: integrated {improved_count} enhanced genomes")

            return improved_count

        except Exception as e:
            logger.warning(f"Quantum self-improvement failed: {e}")
            return 0

    async def federated_average(
        self,
        local_genes: Dict[str, float],
        peer_genes: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Perform federated averaging of gene values.

        Combines local genes with peer genes using weighted average,
        similar to FedAvg for model parameters.

        Args:
            local_genes: Local gene values
            peer_genes: List of gene dicts from peers
            weights: Optional weights for each peer (uniform if None)

        Returns:
            Averaged gene values
        """
        if not peer_genes:
            return local_genes

        all_genes = [local_genes] + peer_genes
        n = len(all_genes)

        if weights is None:
            weights = [1.0 / n] * n
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        # Extend weights to include local
        if len(weights) == len(peer_genes):
            local_weight = 0.5  # Local gets 50% weight
            peer_weight_total = 1.0 - local_weight
            weights = [local_weight] + [w * peer_weight_total for w in weights]

        averaged = {}
        for gene_name in local_genes.keys():
            values = [g.get(gene_name, 0) for g in all_genes]
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            averaged[gene_name] = weighted_sum

        return averaged


# =============================================================================
# INTEGRATION WITH NEXUS FOR EVENT-DRIVEN OPERATION
# =============================================================================

async def setup_federated_evolution(
    genetic_optimizer=None,
    swarm_fabric=None,
    fitness_tracker=None,
    nexus=None,
) -> FederatedPopulationManager:
    """
    Set up federated evolution with Nexus integration.

    Connects the federation manager to Nexus events for:
    - Receiving migration events
    - Receiving fitness reports
    - Broadcasting evolution updates
    """
    manager = FederatedPopulationManager(
        genetic_optimizer=genetic_optimizer,
        swarm_fabric=swarm_fabric,
        fitness_tracker=fitness_tracker,
    )

    if swarm_fabric:
        manager.connect_swarm(swarm_fabric)

    if genetic_optimizer:
        manager.connect_optimizer(genetic_optimizer)

    # Set up Nexus event handlers if available
    if nexus:
        from Cosmos.core.nexus import SignalType

        async def on_genome_migration(signal):
            """Handle incoming genome migration via Nexus."""
            payload = signal.payload
            if payload.get("event") == "genome_migration_received":
                manager.receive_migrant(
                    genome_data=payload.get("genome", {}),
                    fitness_score=payload.get("fitness_score", 0),
                    source_node=payload.get("source_node", "unknown"),
                    generation=payload.get("generation", 0),
                )

        async def on_fitness_received(signal):
            """Handle incoming fitness report via Nexus."""
            payload = signal.payload
            if payload.get("event") == "federated_fitness_received":
                manager.receive_fitness_report(
                    genome_hash=payload.get("genome_hash", ""),
                    fitness_scores=payload.get("fitness", {}),
                    peer_id=payload.get("peer_id", "unknown"),
                )

        nexus.subscribe(SignalType.EXTERNAL_EVENT, on_genome_migration)
        nexus.subscribe(SignalType.EXTERNAL_EVENT, on_fitness_received)

        logger.info("Federated evolution connected to Nexus")

    return manager
