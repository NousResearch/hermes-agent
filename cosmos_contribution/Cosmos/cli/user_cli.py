"""
cosmos Direct-to-User CLI

"Good news, everyone! I've made cosmos accessible to mere mortals!"

A user-friendly command-line interface that provides easy access to all
cosmos features without requiring technical knowledge.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
import json

try:
    import questionary
    from questionary import Style
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False


class MenuCategory(Enum):
    """Main menu categories."""
    HEALTH = "health"
    MEMORY = "memory"
    AGENTS = "agents"
    AUTOMATION = "automation"
    SETTINGS = "settings"
    HELP = "help"


@dataclass
class MenuItem:
    """A menu item configuration."""
    name: str
    description: str
    action: Callable
    category: MenuCategory
    icon: str = ""
    shortcut: Optional[str] = None
    requires_setup: bool = False


class cosmosCLI:
    """
    User-friendly CLI wrapper for cosmos.

    Provides an intuitive interface for:
    - Health tracking and insights
    - Memory operations
    - Agent interactions
    - Workflow automation
    - System configuration
    """

    # Custom style for questionary
    STYLE = Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'fg:white bold'),
        ('answer', 'fg:green bold'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:green'),
        ('separator', 'fg:gray'),
        ('instruction', 'fg:gray italic'),
    ]) if HAS_QUESTIONARY else None

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the user CLI."""
        self.data_dir = Path(data_dir or "./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._memory = None
        self._health_manager = None
        self._n8n = None
        self._initialized = False

        # Build menu items
        self.menu_items: List[MenuItem] = []
        self._build_menu()

    def _build_menu(self):
        """Build the menu structure."""
        # Health menu items
        self.menu_items.extend([
            MenuItem(
                name="View Health Dashboard",
                description="See your health metrics and trends",
                action=self._show_health_dashboard,
                category=MenuCategory.HEALTH,
                icon="",
                shortcut="h",
            ),
            MenuItem(
                name="Log a Meal",
                description="Record what you ate",
                action=self._log_meal,
                category=MenuCategory.HEALTH,
                icon="",
            ),
            MenuItem(
                name="Get Health Tips",
                description="AI-powered health recommendations",
                action=self._get_health_tips,
                category=MenuCategory.HEALTH,
                icon="",
            ),
            MenuItem(
                name="Parse Health Document",
                description="Extract data from lab results or prescriptions",
                action=self._parse_health_document,
                category=MenuCategory.HEALTH,
                icon="",
            ),
            MenuItem(
                name="Set Health Goal",
                description="Create a new health goal",
                action=self._set_health_goal,
                category=MenuCategory.HEALTH,
                icon="",
            ),
        ])

        # Memory menu items
        self.menu_items.extend([
            MenuItem(
                name="Search Memories",
                description="Find something you remembered",
                action=self._search_memories,
                category=MenuCategory.MEMORY,
                icon="",
                shortcut="m",
            ),
            MenuItem(
                name="Save a Memory",
                description="Store something important",
                action=self._save_memory,
                category=MenuCategory.MEMORY,
                icon="",
            ),
            MenuItem(
                name="View Memory Stats",
                description="See how much you've remembered",
                action=self._view_memory_stats,
                category=MenuCategory.MEMORY,
                icon="",
            ),
            MenuItem(
                name="Export Memories",
                description="Export your memories to a file",
                action=self._export_memories,
                category=MenuCategory.MEMORY,
                icon="",
            ),
        ])

        # Agents menu items
        self.menu_items.extend([
            MenuItem(
                name="Ask an Expert",
                description="Get help from a specialist agent",
                action=self._ask_expert,
                category=MenuCategory.AGENTS,
                icon="",
                shortcut="a",
            ),
            MenuItem(
                name="Start a Task",
                description="Delegate a task to agents",
                action=self._start_task,
                category=MenuCategory.AGENTS,
                icon="",
            ),
            MenuItem(
                name="View Active Tasks",
                description="See what agents are working on",
                action=self._view_tasks,
                category=MenuCategory.AGENTS,
                icon="",
            ),
        ])

        # Automation menu items
        self.menu_items.extend([
            MenuItem(
                name="Create Workflow",
                description="Build a new automation workflow",
                action=self._create_workflow,
                category=MenuCategory.AUTOMATION,
                icon="",
                shortcut="w",
            ),
            MenuItem(
                name="Run Workflow",
                description="Execute an existing workflow",
                action=self._run_workflow,
                category=MenuCategory.AUTOMATION,
                icon="",
            ),
            MenuItem(
                name="Connect to n8n",
                description="Link with n8n automation platform",
                action=self._connect_n8n,
                category=MenuCategory.AUTOMATION,
                icon="",
            ),
            MenuItem(
                name="Schedule Task",
                description="Set up a recurring task",
                action=self._schedule_task,
                category=MenuCategory.AUTOMATION,
                icon="",
            ),
        ])

        # Settings menu items
        self.menu_items.extend([
            MenuItem(
                name="Configure Health Providers",
                description="Connect Fitbit, Oura, WHOOP, etc.",
                action=self._configure_health_providers,
                category=MenuCategory.SETTINGS,
                icon="",
            ),
            MenuItem(
                name="System Status",
                description="View system health and connections",
                action=self._system_status,
                category=MenuCategory.SETTINGS,
                icon="",
                shortcut="s",
            ),
            MenuItem(
                name="Preferences",
                description="Customize your experience",
                action=self._preferences,
                category=MenuCategory.SETTINGS,
                icon="",
            ),
        ])

    async def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return

        print("\nInitializing cosmos...")

        # Initialize memory system
        try:
            from Cosmos.memory.memory_system import MemorySystem
            self._memory = MemorySystem(data_dir=str(self.data_dir))
            await self._memory.initialize()
            print("  Memory system ready")
        except Exception as e:
            print(f"  Memory system unavailable: {e}")

        # Initialize health manager
        try:
            from Cosmos.health.providers import HealthProviderManager
            self._health_manager = HealthProviderManager()
            print("  Health tracking ready")
        except Exception as e:
            print(f"  Health tracking unavailable: {e}")

        self._initialized = True
        print("Initialization complete!\n")

    def print_banner(self):
        """Print the welcome banner."""
        banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     cosmos - Your Personal AI Companion               ║
    ║                                                           ║
    ║     "Good news, everyone!"                                ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
        """
        print(banner)

    async def run(self):
        """Run the interactive CLI."""
        self.print_banner()
        await self.initialize()

        if not HAS_QUESTIONARY:
            print("For the best experience, install questionary:")
            print("  pip install questionary")
            await self._run_simple_menu()
            return

        while True:
            try:
                # Main menu
                category = await self._show_main_menu()
                if category is None:
                    break

                # Show category submenu
                await self._show_category_menu(category)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                input("Press Enter to continue...")

    async def _show_main_menu(self) -> Optional[MenuCategory]:
        """Show the main menu and return selected category."""
        choices = [
            questionary.Choice(" Health & Wellness", MenuCategory.HEALTH),
            questionary.Choice(" Memory & Knowledge", MenuCategory.MEMORY),
            questionary.Choice(" AI Agents", MenuCategory.AGENTS),
            questionary.Choice(" Automation", MenuCategory.AUTOMATION),
            questionary.Choice(" Settings", MenuCategory.SETTINGS),
            questionary.Choice(" Exit", None),
        ]

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: questionary.select(
                "What would you like to do?",
                choices=choices,
                style=self.STYLE,
            ).ask()
        )

        return result

    async def _show_category_menu(self, category: MenuCategory):
        """Show submenu for a category."""
        items = [m for m in self.menu_items if m.category == category]

        choices = [
            questionary.Choice(f"{m.icon} {m.name}", m)
            for m in items
        ]
        choices.append(questionary.Choice(" Back to Main Menu", None))

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: questionary.select(
                f"{category.value.title()} Options:",
                choices=choices,
                style=self.STYLE,
            ).ask()
        )

        if result:
            await result.action()

    async def _run_simple_menu(self):
        """Fallback simple menu without questionary."""
        while True:
            print("\n=== Main Menu ===")
            print("1. Health & Wellness")
            print("2. Memory & Knowledge")
            print("3. AI Agents")
            print("4. Automation")
            print("5. Settings")
            print("0. Exit")

            choice = input("\nSelect option: ").strip()

            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                await self._simple_health_menu()
            elif choice == "2":
                await self._simple_memory_menu()
            elif choice == "3":
                await self._simple_agents_menu()
            elif choice == "4":
                await self._simple_automation_menu()
            elif choice == "5":
                await self._simple_settings_menu()

    async def _simple_health_menu(self):
        """Simple health menu."""
        print("\n=== Health & Wellness ===")
        print("1. View Dashboard")
        print("2. Log Meal")
        print("3. Get Health Tips")
        print("4. Parse Document")
        print("0. Back")

        choice = input("Select: ").strip()
        if choice == "1":
            await self._show_health_dashboard()
        elif choice == "2":
            await self._log_meal()
        elif choice == "3":
            await self._get_health_tips()
        elif choice == "4":
            await self._parse_health_document()

    async def _simple_memory_menu(self):
        """Simple memory menu."""
        print("\n=== Memory & Knowledge ===")
        print("1. Search Memories")
        print("2. Save Memory")
        print("3. View Stats")
        print("0. Back")

        choice = input("Select: ").strip()
        if choice == "1":
            await self._search_memories()
        elif choice == "2":
            await self._save_memory()
        elif choice == "3":
            await self._view_memory_stats()

    async def _simple_agents_menu(self):
        """Simple agents menu."""
        print("\n=== AI Agents ===")
        print("1. Ask Expert")
        print("2. Start Task")
        print("3. View Tasks")
        print("0. Back")

        choice = input("Select: ").strip()
        if choice == "1":
            await self._ask_expert()
        elif choice == "2":
            await self._start_task()
        elif choice == "3":
            await self._view_tasks()

    async def _simple_automation_menu(self):
        """Simple automation menu."""
        print("\n=== Automation ===")
        print("1. Create Workflow")
        print("2. Run Workflow")
        print("3. Connect n8n")
        print("0. Back")

        choice = input("Select: ").strip()
        if choice == "1":
            await self._create_workflow()
        elif choice == "2":
            await self._run_workflow()
        elif choice == "3":
            await self._connect_n8n()

    async def _simple_settings_menu(self):
        """Simple settings menu."""
        print("\n=== Settings ===")
        print("1. Health Providers")
        print("2. System Status")
        print("3. Preferences")
        print("0. Back")

        choice = input("Select: ").strip()
        if choice == "1":
            await self._configure_health_providers()
        elif choice == "2":
            await self._system_status()
        elif choice == "3":
            await self._preferences()

    # ========== Health Actions ==========

    async def _show_health_dashboard(self):
        """Display health dashboard."""
        print("\n Health Dashboard")
        print("=" * 50)

        if not self._health_manager:
            print("Health tracking not available.")
            print("Run setup wizard to configure health providers.")
            input("\nPress Enter to continue...")
            return

        try:
            from Cosmos.health.models import MetricType
            from Cosmos.health.analysis import HealthAnalysisEngine

            # Get today's summary
            today = date.today()
            summaries = await self._health_manager.get_daily_summaries(today)

            if summaries:
                summary = summaries[0]
                print(f"\nDate: {today}")
                print(f"  Steps: {summary.total_steps or 'N/A':,}")
                print(f"  Calories: {summary.total_calories_burned or 'N/A'}")
                print(f"  Sleep: {summary.sleep_duration_hours or 'N/A':.1f} hours")
                print(f"  Resting HR: {summary.resting_heart_rate or 'N/A'} bpm")
                print(f"  HRV: {summary.hrv_avg or 'N/A':.1f} ms")

                if summary.sleep_score:
                    print(f"  Sleep Score: {summary.sleep_score}")
                if summary.recovery_score:
                    print(f"  Recovery: {summary.recovery_score}%")
            else:
                print("\nNo health data available for today.")
                print("Connect a health provider in Settings.")

        except Exception as e:
            print(f"Error loading health data: {e}")

        input("\nPress Enter to continue...")

    async def _log_meal(self):
        """Log a meal."""
        print("\n Log a Meal")
        print("=" * 50)

        try:
            from Cosmos.health.nutrition import NutritionManager

            nutrition = NutritionManager(str(self.data_dir / "nutrition"))

            # Get meal type
            meal_types = ["breakfast", "lunch", "dinner", "snack"]
            print("\nMeal type:")
            for i, mt in enumerate(meal_types, 1):
                print(f"  {i}. {mt.title()}")

            choice = input("Select (1-4): ").strip()
            meal_type = meal_types[int(choice) - 1] if choice.isdigit() else "snack"

            # Search for foods
            query = input("\nSearch for food: ").strip()
            results = nutrition.search_foods(query)

            if results:
                print("\nFound foods:")
                for i, food in enumerate(results[:5], 1):
                    print(f"  {i}. {food.name} ({food.calories} cal)")

                selection = input("\nSelect food (number): ").strip()
                if selection.isdigit():
                    food = results[int(selection) - 1]
                    servings = float(input("Servings (default 1): ").strip() or "1")

                    entry = nutrition.log_meal(
                        meal_type=meal_type,
                        foods=[{"food_id": food.id, "servings": servings}],
                    )

                    print(f"\n Logged: {food.name} x{servings}")
                    print(f"   Calories: {food.calories * servings:.0f}")
            else:
                print("No foods found. Try a different search.")

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _get_health_tips(self):
        """Get AI health recommendations."""
        print("\n Health Tips")
        print("=" * 50)

        try:
            from Cosmos.health.swarm_advisor import SwarmHealthAdvisor

            advisor = SwarmHealthAdvisor()

            # Choose focus area
            areas = ["general", "nutrition", "fitness", "sleep", "stress"]
            print("\nFocus area:")
            for i, area in enumerate(areas, 1):
                print(f"  {i}. {area.title()}")

            choice = input("Select (1-5): ").strip()
            focus = areas[int(choice) - 1] if choice.isdigit() else "general"

            print(f"\nGetting {focus} recommendations...")

            # Get recommendations
            recommendations = await advisor.get_recommendations(
                focus_areas=[focus],
                user_profile=None,  # Would load from settings
            )

            print("\n Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"\n{i}. {rec.title}")
                print(f"   {rec.description}")
                if rec.action_items:
                    print("   Actions:")
                    for action in rec.action_items[:3]:
                        print(f"     - {action}")

        except Exception as e:
            print(f"Error getting recommendations: {e}")

        input("\nPress Enter to continue...")

    async def _parse_health_document(self):
        """Parse a health document using OCR."""
        print("\n Parse Health Document")
        print("=" * 50)

        try:
            from Cosmos.health.ocr_parser import DeepSeekOCRParser

            parser = DeepSeekOCRParser()

            # Get document type
            doc_types = ["lab_result", "prescription", "nutrition_label", "medical_report"]
            print("\nDocument type:")
            for i, dt in enumerate(doc_types, 1):
                print(f"  {i}. {dt.replace('_', ' ').title()}")

            choice = input("Select (1-4): ").strip()
            doc_type = doc_types[int(choice) - 1] if choice.isdigit() else "lab_result"

            # Get file path
            file_path = input("\nEnter image path: ").strip()

            if not Path(file_path).exists():
                print("File not found!")
                input("\nPress Enter to continue...")
                return

            print(f"\nParsing {doc_type}...")

            result = await parser.parse_document(file_path, doc_type)

            if result.success:
                print(f"\n Parsed successfully! (Confidence: {result.confidence:.0%})")

                if doc_type == "lab_result" and result.lab_results:
                    print("\nLab Results:")
                    for lab in result.lab_results[:10]:
                        status = "" if lab.in_range else ""
                        print(f"  {status} {lab.name}: {lab.value} {lab.unit}")
                        if lab.reference_range:
                            print(f"      Range: {lab.reference_range}")

                elif doc_type == "prescription" and result.prescriptions:
                    print("\nPrescriptions:")
                    for rx in result.prescriptions:
                        print(f"  {rx.name} {rx.dosage}")
                        print(f"    Instructions: {rx.instructions}")

            else:
                print(f"\nParsing failed: {result.error}")

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _set_health_goal(self):
        """Set a health goal."""
        print("\n Set Health Goal")
        print("=" * 50)

        try:
            from Cosmos.health.models import HealthGoal, GoalType, GoalPeriod

            # Goal type
            goal_types = list(GoalType)
            print("\nGoal type:")
            for i, gt in enumerate(goal_types, 1):
                print(f"  {i}. {gt.value.replace('_', ' ').title()}")

            choice = input("Select: ").strip()
            goal_type = goal_types[int(choice) - 1] if choice.isdigit() else GoalType.STEPS

            # Target value
            target = float(input(f"\nTarget value for {goal_type.value}: ").strip())

            # Period
            print("\nPeriod:")
            print("  1. Daily")
            print("  2. Weekly")
            print("  3. Monthly")

            period_choice = input("Select: ").strip()
            periods = [GoalPeriod.DAILY, GoalPeriod.WEEKLY, GoalPeriod.MONTHLY]
            period = periods[int(period_choice) - 1] if period_choice.isdigit() else GoalPeriod.DAILY

            goal = HealthGoal(
                goal_type=goal_type,
                target_value=target,
                period=period,
            )

            # Save goal
            goals_file = self.data_dir / "health" / "goals.json"
            goals_file.parent.mkdir(parents=True, exist_ok=True)

            goals = []
            if goals_file.exists():
                goals = json.loads(goals_file.read_text())

            goals.append({
                "id": goal.id,
                "type": goal.goal_type.value,
                "target": goal.target_value,
                "period": goal.period.value,
                "created": datetime.now().isoformat(),
            })

            goals_file.write_text(json.dumps(goals, indent=2))

            print(f"\n Goal set: {goal_type.value} - {target} {period.value}")

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    # ========== Memory Actions ==========

    async def _search_memories(self):
        """Search memories."""
        print("\n Search Memories")
        print("=" * 50)

        if not self._memory:
            print("Memory system not available.")
            input("\nPress Enter to continue...")
            return

        query = input("\nWhat are you looking for? ").strip()

        if not query:
            return

        print("\nSearching...")

        try:
            results = await self._memory.recall(query, top_k=5)

            if results:
                print(f"\nFound {len(results)} memories:\n")
                for i, r in enumerate(results, 1):
                    score = f"[{r.score:.0%}]" if hasattr(r, 'score') else ""
                    content = r.content[:200] if len(r.content) > 200 else r.content
                    print(f"{i}. {score} {content}")
                    print()
            else:
                print("No memories found matching your query.")

        except Exception as e:
            print(f"Error searching: {e}")

        input("\nPress Enter to continue...")

    async def _save_memory(self):
        """Save a new memory."""
        print("\n Save Memory")
        print("=" * 50)

        if not self._memory:
            print("Memory system not available.")
            input("\nPress Enter to continue...")
            return

        print("\nWhat would you like to remember?")
        print("(Enter a blank line to finish)\n")

        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        if lines:
            content = "\n".join(lines)

            try:
                mem_id = await self._memory.remember(content)
                print(f"\n Memory saved! (ID: {mem_id[:8]}...)")
            except Exception as e:
                print(f"Error saving: {e}")

        input("\nPress Enter to continue...")

    async def _view_memory_stats(self):
        """View memory statistics."""
        print("\n Memory Statistics")
        print("=" * 50)

        if not self._memory:
            print("Memory system not available.")
            input("\nPress Enter to continue...")
            return

        try:
            stats = self._memory.get_stats()

            print(f"\n Archival Memory")
            archival = stats.get("archival_memory", {})
            print(f"   Total entries: {archival.get('total_entries', 0):,}")
            print(f"   Total size: {archival.get('total_size_mb', 0):.1f} MB")

            print(f"\n Conversation History")
            recall = stats.get("recall_memory", {})
            print(f"   Total turns: {recall.get('total_turns', 0):,}")
            print(f"   Sessions: {recall.get('sessions', 0)}")

            print(f"\n Knowledge Graph")
            kg = stats.get("knowledge_graph", {})
            print(f"   Entities: {kg.get('total_entities', 0):,}")
            print(f"   Relationships: {kg.get('total_relationships', 0):,}")

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _export_memories(self):
        """Export memories to a file."""
        print("\n Export Memories")
        print("=" * 50)

        if not self._memory:
            print("Memory system not available.")
            input("\nPress Enter to continue...")
            return

        output_path = input("\nExport file path (default: memories_export.json): ").strip()
        output_path = output_path or "memories_export.json"

        try:
            from Cosmos.memory.conversation_export import ConversationExporter

            exporter = ConversationExporter(str(self.data_dir))
            await exporter.export_all(output_path, format="json")

            print(f"\n Exported to: {output_path}")

        except Exception as e:
            print(f"Error exporting: {e}")

        input("\nPress Enter to continue...")

    # ========== Agent Actions ==========

    async def _ask_expert(self):
        """Ask a specialist agent."""
        print("\n Ask an Expert")
        print("=" * 50)

        experts = [
            ("Researcher", "research", "Finding and analyzing information"),
            ("Coder", "code", "Programming and debugging"),
            ("Writer", "creative", "Writing and editing"),
            ("Analyst", "reasoning", "Logic and analysis"),
        ]

        print("\nAvailable experts:")
        for i, (name, _, desc) in enumerate(experts, 1):
            print(f"  {i}. {name} - {desc}")

        choice = input("\nSelect expert (1-4): ").strip()

        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(experts):
            return

        expert_name, expert_type, _ = experts[int(choice) - 1]

        question = input(f"\nAsk the {expert_name}: ").strip()

        if not question:
            return

        print(f"\n{expert_name} is thinking...")

        try:
            # Use swarm orchestrator to get response
            from Cosmos.agents.swarm_orchestrator import SwarmOrchestrator

            orchestrator = SwarmOrchestrator()

            response = await orchestrator.process(
                query=question,
                task_type=expert_type,
            )

            print(f"\n{expert_name}'s Response:")
            print("-" * 40)
            print(response.get("response", "No response generated."))

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _start_task(self):
        """Start a task with agents."""
        print("\n Start a Task")
        print("=" * 50)

        print("\nDescribe your task:")
        task = input("> ").strip()

        if not task:
            return

        print("\nAssigning task to agents...")

        try:
            from Cosmos.agents.swarm_orchestrator import SwarmOrchestrator

            orchestrator = SwarmOrchestrator()
            result = await orchestrator.process(query=task)

            print("\n Task completed!")
            print("-" * 40)
            print(result.get("response", "Task processed."))

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _view_tasks(self):
        """View active tasks."""
        print("\n Active Tasks")
        print("=" * 50)

        # This would integrate with a task tracking system
        print("\nNo active tasks at the moment.")
        print("Start a new task from the Agents menu.")

        input("\nPress Enter to continue...")

    # ========== Automation Actions ==========

    async def _create_workflow(self):
        """Create a new workflow."""
        print("\n Create Workflow")
        print("=" * 50)

        try:
            from Cosmos.automation.workflow_builder import WorkflowBuilder

            builder = WorkflowBuilder()

            name = input("\nWorkflow name: ").strip()
            if not name:
                return

            description = input("Description: ").strip()

            builder.create_workflow(name, description)

            # Add steps
            print("\nAdd workflow steps (type 'done' when finished):")
            print("Available actions: notify, log_health, run_query, trigger_n8n, send_email")

            while True:
                step = input("\nStep action (or 'done'): ").strip()
                if step.lower() == 'done':
                    break

                params = input("Parameters (JSON or key=value): ").strip()

                builder.add_step(step, params)
                print(f" Added: {step}")

            # Save workflow
            workflow_id = builder.save()
            print(f"\n Workflow created: {workflow_id}")

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _run_workflow(self):
        """Run an existing workflow."""
        print("\n Run Workflow")
        print("=" * 50)

        try:
            from Cosmos.automation.workflow_builder import WorkflowBuilder

            builder = WorkflowBuilder()
            workflows = builder.list_workflows()

            if not workflows:
                print("No workflows found. Create one first!")
                input("\nPress Enter to continue...")
                return

            print("\nAvailable workflows:")
            for i, wf in enumerate(workflows, 1):
                print(f"  {i}. {wf['name']} - {wf.get('description', 'No description')}")

            choice = input("\nSelect workflow: ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(workflows):
                workflow = workflows[int(choice) - 1]

                print(f"\nRunning {workflow['name']}...")

                result = await builder.execute(workflow['id'])

                print(f"\n Workflow completed!")
                print(f"   Steps run: {result.get('steps_completed', 0)}")
                print(f"   Duration: {result.get('duration_ms', 0)}ms")

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _connect_n8n(self):
        """Connect to n8n."""
        print("\n Connect to n8n")
        print("=" * 50)

        try:
            from Cosmos.automation.n8n_enhanced import EnhancedN8nIntegration

            print("\nn8n is a powerful workflow automation tool.")
            print("You can connect cosmos to trigger and receive n8n workflows.\n")

            base_url = input("n8n URL (e.g., http://localhost:5678): ").strip()
            api_key = input("n8n API Key: ").strip()

            if not base_url or not api_key:
                print("URL and API key required.")
                input("\nPress Enter to continue...")
                return

            n8n = EnhancedN8nIntegration(api_key=api_key, base_url=base_url)
            connected = await n8n.connect()

            if connected:
                print("\n Connected to n8n!")

                # List workflows
                workflows = await n8n.list_workflows()
                if workflows:
                    print(f"\nFound {len(workflows)} workflows:")
                    for wf in workflows[:5]:
                        print(f"  - {wf.get('name', 'Unnamed')}")
            else:
                print(" Failed to connect. Check your credentials.")

        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")

    async def _schedule_task(self):
        """Schedule a recurring task."""
        print("\n Schedule Task")
        print("=" * 50)

        print("\nSchedule types:")
        print("  1. Every X minutes")
        print("  2. Daily at specific time")
        print("  3. Weekly on specific day")

        choice = input("\nSelect schedule type: ").strip()

        task_name = input("Task name: ").strip()
        task_action = input("Task action: ").strip()

        schedule_config = {}

        if choice == "1":
            minutes = input("Every X minutes: ").strip()
            schedule_config = {"type": "interval", "minutes": int(minutes)}
        elif choice == "2":
            time = input("Time (HH:MM): ").strip()
            schedule_config = {"type": "daily", "time": time}
        elif choice == "3":
            day = input("Day (mon/tue/wed/thu/fri/sat/sun): ").strip()
            time = input("Time (HH:MM): ").strip()
            schedule_config = {"type": "weekly", "day": day, "time": time}

        # Save schedule
        schedules_file = self.data_dir / "schedules.json"
        schedules = []
        if schedules_file.exists():
            schedules = json.loads(schedules_file.read_text())

        schedules.append({
            "name": task_name,
            "action": task_action,
            "schedule": schedule_config,
            "created": datetime.now().isoformat(),
            "enabled": True,
        })

        schedules_file.write_text(json.dumps(schedules, indent=2))

        print(f"\n Task scheduled: {task_name}")

        input("\nPress Enter to continue...")

    # ========== Settings Actions ==========

    async def _configure_health_providers(self):
        """Configure health providers."""
        print("\n Configure Health Providers")
        print("=" * 50)

        providers = [
            ("Fitbit", "fitbit", "FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET"),
            ("Oura Ring", "oura", "OURA_ACCESS_TOKEN"),
            ("WHOOP", "whoop", "WHOOP_CLIENT_ID, WHOOP_CLIENT_SECRET"),
            ("Apple Health", "apple_health", "APPLE_HEALTH_EXPORT (path to export.xml)"),
        ]

        print("\nSupported providers:")
        for name, key, env_vars in providers:
            print(f"  - {name}")
            print(f"    Required: {env_vars}")
            print()

        print("To configure providers, add the required environment variables")
        print("to your .env file or system environment.")

        input("\nPress Enter to continue...")

    async def _system_status(self):
        """Show system status."""
        print("\n System Status")
        print("=" * 50)

        # Memory
        memory_status = "" if self._memory else ""
        print(f"\n{memory_status} Memory System")
        if self._memory:
            stats = self._memory.get_stats()
            print(f"   Entries: {stats.get('archival_memory', {}).get('total_entries', 0):,}")

        # Health
        health_status = "" if self._health_manager else ""
        print(f"\n{health_status} Health Tracking")
        if self._health_manager:
            providers = await self._health_manager.get_connected_providers()
            print(f"   Providers: {len(providers)}")

        # Check for various integrations
        try:
            from Cosmos.core.swarm.p2p import swarm_fabric
            p2p_status = "" if swarm_fabric.peers else ""
            print(f"\n{p2p_status} P2P Network")
            print(f"   Peers: {len(swarm_fabric.peers)}")
        except:
            print(f"\n P2P Network (not running)")

        input("\nPress Enter to continue...")

    async def _preferences(self):
        """Configure preferences."""
        print("\n Preferences")
        print("=" * 50)

        prefs_file = self.data_dir / "preferences.json"

        # Load existing preferences
        prefs = {}
        if prefs_file.exists():
            prefs = json.loads(prefs_file.read_text())

        print("\nCurrent preferences:")
        print(f"  1. Theme: {prefs.get('theme', 'default')}")
        print(f"  2. Notifications: {prefs.get('notifications', True)}")
        print(f"  3. Auto-backup: {prefs.get('auto_backup', True)}")
        print(f"  4. Language: {prefs.get('language', 'en')}")

        print("\n0. Save and exit")

        while True:
            choice = input("\nEdit preference (0-4): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                prefs['theme'] = input("Theme (default/dark/light): ").strip() or "default"
            elif choice == "2":
                prefs['notifications'] = input("Notifications (yes/no): ").strip().lower() == "yes"
            elif choice == "3":
                prefs['auto_backup'] = input("Auto-backup (yes/no): ").strip().lower() == "yes"
            elif choice == "4":
                prefs['language'] = input("Language code (en/es/fr/de): ").strip() or "en"

        # Save preferences
        prefs_file.write_text(json.dumps(prefs, indent=2))
        print("\n Preferences saved!")

        input("\nPress Enter to continue...")


async def run_user_cli(data_dir: Optional[str] = None):
    """Entry point for the user CLI."""
    cli = cosmosCLI(data_dir=data_dir)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(run_user_cli())
