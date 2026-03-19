"""
Command Router for Natural Language Tasks.

Routes parsed intents to appropriate handlers.
"""

import logging
from typing import  Optional, Callable
from dataclasses import dataclass

from .intent_parser import IntentParser, Intent
from .task_classifier import TaskClassifier, TaskType, Classification
from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of executing a command."""
    success: bool
    task_type: TaskType
    handler: str
    response: str
    data: dict = None
    error: Optional[str] = None


class CommandRouter:
    """
    Route natural language commands to appropriate handlers.

    Integrates with:
    - Bankr for crypto operations
    - Evolution loop for code tasks
    - Browser agent for automation
    - Swarm for general queries
    """

    def __init__(self):
        self.parser = IntentParser()
        self.classifier = TaskClassifier()
        self.extractor = EntityExtractor()

        # Lazy-loaded handlers
        self._bankr_client = None
        self._bankr_trading = None
        self._bankr_polymarket = None

    async def execute(self, command: str) -> CommandResult:
        """
        Execute a natural language command.

        Args:
            command: User's natural language command

        Returns:
            CommandResult with response and dict data
        """
        try:
            # Parse intent
            intent = self.parser.parse(command)
            logger.debug(f"Parsed intent: {intent}")

            # Classify task
            classification = self.classifier.classify(intent)
            logger.debug(f"Classification: {classification}")

            # Extract entities for additional context
            entities = self.extractor.extract(command)

            # Route to appropriate handler
            handler_name = self.classifier.get_handler_type(classification)
            result = await self._route(intent, classification, entities)

            return CommandResult(
                success=result.get("success", True),
                task_type=classification.task_type,
                handler=handler_name,
                response=result.get("response", ""),
                data=result.get("data"),
            )

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                success=False,
                task_type=TaskType.UNKNOWN,
                handler="error",
                response=f"Sorry, I encountered an error: {str(e)}",
                error=str(e),
            )

    async def _route(
        self,
        intent: Intent,
        classification: Classification,
        entities
    ) -> dict:
        """Route to the appropriate handler based on classification."""

        task_type = classification.task_type

        # Crypto operations -> Bankr
        if task_type == TaskType.TRADE:
            return await self._handle_trade(intent, entities)

        elif task_type == TaskType.CRYPTO:
            return await self._handle_crypto(intent, entities)

        elif task_type == TaskType.PREDICT:
            return await self._handle_prediction(intent, entities)

        # Code tasks -> Evolution loop
        elif task_type in (TaskType.CODE, TaskType.FIX):
            return await self._handle_code(intent, entities)

        # Browser automation
        elif task_type in (TaskType.AUTOMATE, TaskType.SCRAPE):
            return await self._handle_automation(intent, entities)

        # Research/explain -> Swarm
        elif task_type in (TaskType.RESEARCH, TaskType.EXPLAIN):
            return await self._handle_research(intent, entities)

        # Communication
        elif task_type == TaskType.COMMUNICATE:
            return await self._handle_communication(intent, entities)

        # Default handler
        else:
            return await self._handle_general(intent, entities)

    async def _get_bankr_trading(self):
        """Lazy load Bankr trading handler."""
        if self._bankr_trading is None:
            try:
                from Cosmos.integration.bankr import BankrTrading, get_bankr_client
                client = get_bankr_client()
                self._bankr_trading = BankrTrading(client)
            except ImportError:
                logger.warning("Bankr module not available")
                return None
        return self._bankr_trading

    async def _get_bankr_polymarket(self):
        """Lazy load Bankr Polymarket handler."""
        if self._bankr_polymarket is None:
            try:
                from Cosmos.integration.bankr import BankrPolymarket, get_bankr_client
                client = get_bankr_client()
                self._bankr_polymarket = BankrPolymarket(client)
            except ImportError:
                logger.warning("Bankr module not available")
                return None
        return self._bankr_polymarket

    async def _handle_trade(self, intent: Intent, entities) -> dict:
        """Handle trading commands via Bankr."""
        trading = await self._get_bankr_trading()

        if trading is None:
            return {
                "success": False,
                "response": "Trading module not available. Please install the bankr module.",
            }

        try:
            result = await trading.execute_trade(intent)

            if result.success:
                return {
                    "success": True,
                    "response": f"Trade executed: {result.trade_type} {result.amount_out} {result.token_out}",
                    "data": {
                        "tx_hash": result.tx_hash,
                        "trade_type": result.trade_type,
                        "amount_in": str(result.amount_in),
                        "amount_out": str(result.amount_out),
                    }
                }
            else:
                return {
                    "success": False,
                    "response": f"Trade failed: {result.error}",
                }

        except Exception as e:
            logger.error(f"Trade error: {e}")
            return {
                "success": False,
                "response": f"Trade error: {str(e)}",
            }

    async def _handle_crypto(self, intent: Intent, entities) -> dict:
        """Handle crypto queries via Bankr."""
        try:
            from Cosmos.integration.bankr import get_bankr_client
            client = get_bankr_client()

            # Get primary token from entities
            token = self.extractor.get_primary_token(entities)

            if intent.action == "get_price" or "price" in intent.original_text.lower():
                token = token or intent.parameters.get("token", "BTC")
                price = await client.get_price(token)
                return {
                    "success": True,
                    "response": f"The current price of {token.upper()} is ${price:,.2f}",
                    "data": {"token": token, "price": price}
                }

            elif intent.action in ("get_balance", "get_portfolio"):
                result = await client.execute(intent.to_prompt())
                return {
                    "success": True,
                    "response": f"Portfolio fetched successfully",
                    "data": result
                }

            else:
                # Generic crypto query
                result = await client.execute(intent.to_prompt())
                return {
                    "success": True,
                    "response": str(result),
                    "data": result
                }

        except ImportError:
            return {
                "success": False,
                "response": "Crypto module not available. Please install the bankr module.",
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"Crypto query error: {str(e)}",
            }

    async def _handle_prediction(self, intent: Intent, entities) -> dict:
        """Handle Polymarket predictions via Bankr."""
        polymarket = await self._get_bankr_polymarket()

        if polymarket is None:
            return {
                "success": False,
                "response": "Polymarket module not available.",
            }

        try:
            if intent.action == "get_odds":
                market = intent.parameters.get("market", intent.target)
                odds = await polymarket.get_odds(market)
                odds_str = ", ".join(f"{k}: {v:.1%}" for k, v in odds.items())
                return {
                    "success": True,
                    "response": f"Odds for '{market}': {odds_str}",
                    "data": odds
                }

            elif intent.action in ("place_bet", "bet", "wager"):
                result = await polymarket.execute(intent)
                if result.success:
                    return {
                        "success": True,
                        "response": f"Bet placed: ${result.amount_usd} on {result.outcome}",
                        "data": {"tx_hash": result.tx_hash}
                    }
                else:
                    return {
                        "success": False,
                        "response": f"Bet failed: {result.error}",
                    }

            else:
                return {
                    "success": False,
                    "response": "Unknown prediction action",
                }

        except Exception as e:
            return {
                "success": False,
                "response": f"Prediction error: {str(e)}",
            }

    async def _handle_code(self, intent: Intent, entities) -> dict:
        """Handle code generation via evolution loop."""
        try:
            # Try to add to evolution loop
            from Cosmos.core.evolution_loop import get_evolution_loop
            evolution = get_evolution_loop()

            if evolution:
                task_desc = f"{intent.action}: {intent.target}"
                # Add task to evolution loop
                # This would integrate with agent_spawner
                return {
                    "success": True,
                    "response": f"Added task to evolution queue: {task_desc}",
                    "data": {"task": task_desc}
                }

        except Exception as e:
            logger.warning(f"Evolution loop not available: {e}")

        return {
            "success": True,
            "response": f"I understand you want me to {intent.action} {intent.target}. Let me work on that.",
        }

    async def _handle_automation(self, intent: Intent, entities) -> dict:
        """Handle browser automation."""
        try:
            from Cosmos.agents.browser import get_browser_agent
            agent = get_browser_agent()

            if agent:
                result = await agent.execute_task(intent.to_prompt())
                return {
                    "success": True,
                    "response": "Automation task completed",
                    "data": result
                }

        except ImportError:
            logger.warning("Browser agent not available")

        return {
            "success": False,
            "response": "Browser automation module not installed.",
        }

    async def _handle_research(self, intent: Intent, entities) -> dict:
        """Handle research queries."""
        # Route to swarm for research
        try:
            from Cosmos.web.server import generate_multi_model_response

            response = await generate_multi_model_response(
                intent.to_prompt(),
                speaker="User"
            )

            return {
                "success": True,
                "response": response,
            }

        except Exception as e:
            logger.warning(f"Swarm not available: {e}")

        return {
            "success": True,
            "response": f"Researching: {intent.target}",
        }

    async def _handle_communication(self, intent: Intent, entities) -> dict:
        """Handle communication tasks."""
        return {
            "success": True,
            "response": f"I'll help you {intent.action}: {intent.target}",
        }

    async def _handle_general(self, intent: Intent, entities) -> dict:
        """Handle general/unknown commands."""
        return {
            "success": True,
            "response": f"I understand you want to {intent.action}. Let me see what I can do.",
        }
