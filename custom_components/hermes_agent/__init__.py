"""Hermes Agent Home Assistant Integration - Main module."""

import asyncio
import logging
from typing import Final

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    DOMAIN,
    SERVICE_CHAT,
    SERVICE_EXECUTE_SKILL,
    SERVICE_QUERY,
    ATTR_MESSAGE,
    ATTR_SKILL_NAME,
    ATTR_ARGS,
    CONF_GATEWAY_HOST,
    CONF_GATEWAY_PORT,
)
from .coordinator import HermesAgentCoordinator

_LOGGER = logging.getLogger(__name__)
PLATFORMS: Final = ["sensor"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Hermes Agent from a config entry."""
    coordinator = HermesAgentCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    async def handle_chat(call: ServiceCall) -> None:
        """Handle chat service call."""
        message = call.data.get(ATTR_MESSAGE)
        await _send_to_gateway(
            hass, entry, "chat", {ATTR_MESSAGE: message}
        )

    async def handle_execute_skill(call: ServiceCall) -> None:
        """Handle execute skill service call."""
        skill_name = call.data.get(ATTR_SKILL_NAME)
        args = call.data.get(ATTR_ARGS, {})
        await _send_to_gateway(
            hass, entry, "execute_skill", {
                ATTR_SKILL_NAME: skill_name,
                ATTR_ARGS: args,
            }
        )

    async def handle_query(call: ServiceCall) -> None:
        """Handle query service call."""
        message = call.data.get(ATTR_MESSAGE)
        await _send_to_gateway(
            hass, entry, "query", {ATTR_MESSAGE: message}
        )

    hass.services.async_register(DOMAIN, SERVICE_CHAT, handle_chat)
    hass.services.async_register(DOMAIN, SERVICE_EXECUTE_SKILL, handle_execute_skill)
    hass.services.async_register(DOMAIN, SERVICE_QUERY, handle_query)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(
        entry, PLATFORMS
    ):
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok


async def _send_to_gateway(
    hass: HomeAssistant, entry: ConfigEntry, endpoint: str, payload: dict
) -> None:
    """Send request to Hermes Agent gateway."""
    try:
        host = entry.data[CONF_GATEWAY_HOST]
        port = entry.data[CONF_GATEWAY_PORT]
        url = f"http://{host}:{port}/{endpoint}"

        session = async_get_clientsession(hass)
        async with asyncio.timeout(30):
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    _LOGGER.error(f"Gateway error: {resp.status}")
                else:
                    _LOGGER.debug(f"Gateway response: {resp.status}")
    except asyncio.TimeoutError:
        _LOGGER.error("Timeout connecting to gateway")
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.error(f"Error sending to gateway: {err}")
