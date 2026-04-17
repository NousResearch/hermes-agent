"""Tests para precisión de routing de Opus."""
import os
import pytest

from tests.routing.veracity._helpers import load_activation, load_post_activation_events


class TestOpusRoutingPrecision:
    """
    Valida que:
    - Eventos con turn_kind opus_keyword/continuation NO usen mini
    - Eventos simples con mini NO tengan marcadores de opus
    """

    def test_opus_keyword_not_routed_to_mini(self):
        """
        Eventos con turn_kind en {opus_keyword, continuation} NO deben tener model con 'mini'.
        """
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para determinar cutoff")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        events = load_post_activation_events(activation=activation)
        
        if len(events) < 20:
            pytest.skip(f"Menos de 20 eventos post-activación ({len(events)})")
        
        opus_keywords = {"opus_keyword", "continuation"}
        contaminados = []
        
        for event in events:
            turn_kind = event.get("turn_kind", "")
            model = event.get("model", "").lower()
            
            if turn_kind in opus_keywords:
                if "mini" in model or "gpt-5-mini" in model:
                    contaminados.append({
                        "timestamp": event.get("timestamp"),
                        "model": event.get("model"),
                        "turn_kind": turn_kind
                    })
        
        assert len(contaminados) == 0, f"{len(contaminados)} eventos opus contaminados con mini: {contaminados[:5]}"

    def test_simple_mini_no_opus_markers(self):
        """
        Eventos con turn_kind=='simple' y model con 'mini' NO deben tener marcadores [OPUS]/[HEAVY]/etc.
        
        Nota: Si no guardamos user_message_hash o tags en el evento, este test pasa por vacuidad.
        """
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para determinar cutoff")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        events = load_post_activation_events(activation=activation)
        
        if len(events) < 20:
            pytest.skip(f"Menos de 20 eventos post-activación ({len(events)})")
        
        opus_markers = {"[OPUS]", "[HEAVY]", "[CRÍTICO]", "[ARCHITECTURE]", "[CRITICO]"}
        violaciones = []
        
        for event in events:
            turn_kind = event.get("turn_kind", "")
            model = event.get("model", "").lower()
            
            if turn_kind == "simple" and "mini" in model:
                # Buscar marcadores en user_message si existe
                user_msg = event.get("user_message", "")
                tags = event.get("tags", [])
                
                content_to_check = user_msg
                if isinstance(tags, list):
                    content_to_check += " " + " ".join(str(t) for t in tags)
                
                for marker in opus_markers:
                    if marker.lower() in content_to_check.lower():
                        violaciones.append({
                            "timestamp": event.get("timestamp"),
                            "model": event.get("model"),
                            "marker_found": marker
                        })
                        break
        
        # Documentar que este test puede pasar por vacuidad si no hay marcadores guardados
        if len(violaciones) == 0:
            # Verificar si hay eventos simples con mini para confirmar que el test hizo algo
            simple_mini_count = sum(
                1 for e in events 
                if e.get("turn_kind") == "simple" and "mini" in e.get("model", "").lower()
            )
            if simple_mini_count == 0:
                pytest.skip("No hay eventos 'simple' con mini para validar (test pasa por vacuidad)")
        
        assert len(violaciones) == 0, f"{len(violaciones)} violaciones: {violaciones[:5]}"
