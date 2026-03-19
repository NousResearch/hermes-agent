"""
cosmos Behavior Mutation - Swarm Behavior Evolution

Novel Approaches:
1. Behavioral Genome - Encode swarm behaviors as genes
2. Team Composition Evolution - Optimize agent team structures
3. Strategy Discovery - Emergent behavior through mutation
4. Multi-Objective Balancing - Balance speed, quality, efficiency
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from collections import defaultdict

from loguru import logger


@dataclass
class BehaviorGene:
    """A behavioral parameter."""
    name: str
    value: float
    min_val: float
    max_val: float

    def mutate(self, sigma: float = 0.1) -> "BehaviorGene":
        """Create mutated copy."""
        delta = random.gauss(0, sigma * (self.max_val - self.min_val))
        new_value = max(self.min_val, min(self.max_val, self.value + delta))
        return BehaviorGene(
            name=self.name,
            value=new_value,
            min_val=self.min_val,
            max_val=self.max_val,
        )


@dataclass
class BehaviorGenome:
    """Complete behavioral genome for an agent or swarm."""
    id: str
    genes: dict[str, BehaviorGene]
    generation: int = 0
    fitness: float = 0.0
    uses: int = 0

    def get(self, gene_name: str, default: float = 0.5) -> float:
        """Get gene value."""
        return self.genes[gene_name].value if gene_name in self.genes else default

    def to_dict(self) -> dict:
        """Serialize genome."""
        return {
            "id": self.id,
            "genes": {name: gene.value for name, gene in self.genes.items()},
            "generation": self.generation,
            "fitness": self.fitness,
            "uses": self.uses,
        }


@dataclass
class TeamComposition:
    """A team of agents with composition parameters."""
    id: str
    agent_counts: dict[str, int]  # agent_type -> count
    coordination_style: str  # "parallel", "sequential", "hierarchical"
    communication_level: float  # 0-1, how much agents share
    specialization_level: float  # 0-1, how specialized vs generalist

    # Performance
    fitness: float = 0.0
    uses: int = 0
    successes: int = 0


class BehaviorMutator:
    """
    Evolves behavioral parameters for agents and swarms.

    Features:
    - Behavioral genome encoding
    - Team composition optimization
    - Strategy mutation and discovery
    - Performance-based selection
    """

    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

        # Behavior genomes
        self.behavior_population: list[BehaviorGenome] = []
        self.active_behavior: Optional[BehaviorGenome] = None

        # Team compositions
        self.team_population: list[TeamComposition] = []
        self.active_team: Optional[TeamComposition] = None

        # Default gene definitions
        self.gene_definitions = {
            # Communication
            "communication_frequency": (0.1, 1.0, 0.5),
            "information_sharing": (0.2, 0.9, 0.6),

            # Specialization
            "specialization_factor": (0.3, 0.9, 0.7),
            "handoff_threshold": (0.4, 0.9, 0.65),

            # Cooperation
            "cooperation_level": (0.2, 0.8, 0.5),
            "independence_factor": (0.2, 0.8, 0.5),

            # Exploration
            "exploration_rate": (0.05, 0.3, 0.1),
            "novelty_seeking": (0.1, 0.5, 0.2),

            # Resource usage
            "token_budget_factor": (0.5, 1.5, 1.0),
            "time_budget_factor": (0.5, 1.5, 1.0),

            # Quality vs speed
            "quality_priority": (0.3, 0.9, 0.6),
            "speed_priority": (0.3, 0.9, 0.5),
        }

        # Agent types for team composition
        self.agent_types = ["code", "reasoning", "research", "creative", "meta"]

        self.generation = 0
        self.stats = {
            "generations_evolved": 0,
            "behaviors_evaluated": 0,
            "teams_evaluated": 0,
        }

    def initialize_population(self):
        """Initialize behavior population."""
        self.behavior_population = []

        # Create default genome
        default = self._create_default_genome()
        self.behavior_population.append(default)
        self.active_behavior = default

        # Fill with random variations
        while len(self.behavior_population) < self.population_size:
            genome = self._create_random_genome()
            self.behavior_population.append(genome)

        # Initialize team compositions
        self._initialize_teams()

        logger.info(f"Initialized {len(self.behavior_population)} behavior genomes")

    def _create_default_genome(self) -> BehaviorGenome:
        """Create genome with default values."""
        genes = {}
        for name, (min_val, max_val, default) in self.gene_definitions.items():
            genes[name] = BehaviorGene(
                name=name,
                value=default,
                min_val=min_val,
                max_val=max_val,
            )

        return BehaviorGenome(
            id=f"behavior_default",
            genes=genes,
            generation=0,
        )

    def _create_random_genome(self) -> BehaviorGenome:
        """Create genome with random values."""
        genes = {}
        for name, (min_val, max_val, _) in self.gene_definitions.items():
            value = random.uniform(min_val, max_val)
            genes[name] = BehaviorGene(
                name=name,
                value=value,
                min_val=min_val,
                max_val=max_val,
            )

        return BehaviorGenome(
            id=f"behavior_g{self.generation}_{random.randint(0, 9999):04d}",
            genes=genes,
            generation=self.generation,
        )

    def _initialize_teams(self):
        """Initialize team composition population."""
        self.team_population = []

        # Default balanced team
        default_team = TeamComposition(
            id="team_balanced",
            agent_counts={t: 1 for t in self.agent_types},
            coordination_style="parallel",
            communication_level=0.5,
            specialization_level=0.7,
        )
        self.team_population.append(default_team)
        self.active_team = default_team

        # Minimal team
        self.team_population.append(TeamComposition(
            id="team_minimal",
            agent_counts={"code": 1, "reasoning": 1},
            coordination_style="sequential",
            communication_level=0.8,
            specialization_level=0.9,
        ))

        # Large team
        self.team_population.append(TeamComposition(
            id="team_large",
            agent_counts={t: 2 for t in self.agent_types},
            coordination_style="hierarchical",
            communication_level=0.6,
            specialization_level=0.6,
        ))

    def mutate_genome(self, genome: BehaviorGenome) -> BehaviorGenome:
        """Create mutated copy of a genome."""
        new_genes = {}
        for name, gene in genome.genes.items():
            if random.random() < self.mutation_rate:
                new_genes[name] = gene.mutate()
            else:
                new_genes[name] = BehaviorGene(
                    name=gene.name,
                    value=gene.value,
                    min_val=gene.min_val,
                    max_val=gene.max_val,
                )

        return BehaviorGenome(
            id=f"behavior_g{self.generation}_{random.randint(0, 9999):04d}",
            genes=new_genes,
            generation=self.generation,
        )

    def crossover_genomes(
        self,
        parent1: BehaviorGenome,
        parent2: BehaviorGenome,
    ) -> BehaviorGenome:
        """Create offspring from two parents."""
        new_genes = {}
        for name in self.gene_definitions.keys():
            # Randomly select from either parent
            source = parent1 if random.random() < 0.5 else parent2
            gene = source.genes[name]
            new_genes[name] = BehaviorGene(
                name=gene.name,
                value=gene.value,
                min_val=gene.min_val,
                max_val=gene.max_val,
            )

        return BehaviorGenome(
            id=f"behavior_g{self.generation}_{random.randint(0, 9999):04d}",
            genes=new_genes,
            generation=self.generation,
        )

    def mutate_team(self, team: TeamComposition) -> TeamComposition:
        """Create mutated copy of a team composition."""
        new_counts = dict(team.agent_counts)

        # Randomly adjust agent counts
        for agent_type in self.agent_types:
            if random.random() < self.mutation_rate:
                delta = random.choice([-1, 0, 1])
                new_counts[agent_type] = max(0, new_counts.get(agent_type, 0) + delta)

        # Ensure at least one agent
        if sum(new_counts.values()) == 0:
            new_counts[random.choice(self.agent_types)] = 1

        # Mutate other parameters
        new_comm = team.communication_level
        new_spec = team.specialization_level
        if random.random() < self.mutation_rate:
            new_comm = max(0.1, min(0.9, new_comm + random.gauss(0, 0.1)))
        if random.random() < self.mutation_rate:
            new_spec = max(0.1, min(0.9, new_spec + random.gauss(0, 0.1)))

        return TeamComposition(
            id=f"team_g{self.generation}_{random.randint(0, 9999):04d}",
            agent_counts=new_counts,
            coordination_style=random.choice(["parallel", "sequential", "hierarchical"]),
            communication_level=new_comm,
            specialization_level=new_spec,
        )

    def record_behavior_result(
        self,
        genome_id: str,
        success: bool,
        metrics: dict[str, float],
    ):
        """Record result for a behavior genome."""
        for genome in self.behavior_population:
            if genome.id == genome_id:
                genome.uses += 1

                # Update fitness based on metrics
                quality = metrics.get("quality", 0.5)
                efficiency = metrics.get("efficiency", 0.5)
                success_bonus = 1.0 if success else 0.0

                new_fitness = (quality * 0.4 + efficiency * 0.3 + success_bonus * 0.3)
                genome.fitness = genome.fitness * 0.9 + new_fitness * 0.1

                self.stats["behaviors_evaluated"] += 1
                break

    def record_team_result(
        self,
        team_id: str,
        success: bool,
        metrics: dict[str, float],
    ):
        """Record result for a team composition."""
        for team in self.team_population:
            if team.id == team_id:
                team.uses += 1
                team.successes += int(success)

                quality = metrics.get("quality", 0.5)
                efficiency = metrics.get("efficiency", 0.5)
                success_bonus = 1.0 if success else 0.0

                new_fitness = (quality * 0.4 + efficiency * 0.3 + success_bonus * 0.3)
                team.fitness = team.fitness * 0.9 + new_fitness * 0.1

                self.stats["teams_evaluated"] += 1
                break

    def evolve_generation(self):
        """Evolve one generation of behaviors and teams."""
        self.generation += 1

        # Evolve behaviors
        self.behavior_population.sort(key=lambda g: g.fitness, reverse=True)

        # Keep top performers
        survivors = self.behavior_population[:self.population_size // 2]

        # Create offspring
        while len(survivors) < self.population_size:
            if random.random() < 0.7 and len(survivors) >= 2:
                # Crossover
                p1, p2 = random.sample(survivors[:5], 2)
                child = self.crossover_genomes(p1, p2)
            else:
                # Mutation
                parent = random.choice(survivors[:5])
                child = self.mutate_genome(parent)

            survivors.append(child)

        self.behavior_population = survivors
        self.active_behavior = self.behavior_population[0]

        # Evolve teams
        self.team_population.sort(key=lambda t: t.fitness, reverse=True)

        team_survivors = self.team_population[:3]
        while len(team_survivors) < 5:
            parent = random.choice(team_survivors[:2])
            child = self.mutate_team(parent)
            team_survivors.append(child)

        self.team_population = team_survivors
        self.active_team = self.team_population[0]

        self.stats["generations_evolved"] += 1

        logger.info(f"Evolved to generation {self.generation}")

    def select_behavior(self) -> BehaviorGenome:
        """Select a behavior for use (with exploration)."""
        if random.random() < 0.1:
            # Exploration
            return random.choice(self.behavior_population)
        return self.active_behavior or self.behavior_population[0]

    def select_team(self) -> TeamComposition:
        """Select a team composition for use."""
        if random.random() < 0.1:
            return random.choice(self.team_population)
        return self.active_team or self.team_population[0]

    def get_behavior_params(self) -> dict[str, float]:
        """Get current active behavior parameters."""
        if self.active_behavior:
            return {name: gene.value for name, gene in self.active_behavior.genes.items()}
        return {}

    def get_team_config(self) -> dict:
        """Get current active team configuration."""
        if self.active_team:
            return {
                "agent_counts": self.active_team.agent_counts,
                "coordination_style": self.active_team.coordination_style,
                "communication_level": self.active_team.communication_level,
                "specialization_level": self.active_team.specialization_level,
            }
        return {}

    def get_stats(self) -> dict:
        """Get mutator statistics."""
        return {
            **self.stats,
            "generation": self.generation,
            "behavior_count": len(self.behavior_population),
            "team_count": len(self.team_population),
            "active_behavior": self.active_behavior.id if self.active_behavior else None,
            "active_team": self.active_team.id if self.active_team else None,
            "top_behaviors": [
                {"id": g.id, "fitness": g.fitness}
                for g in sorted(self.behavior_population, key=lambda x: x.fitness, reverse=True)[:3]
            ],
        }
