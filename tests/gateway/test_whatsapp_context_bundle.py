from __future__ import annotations

from pathlib import Path

from gateway.whatsapp_context import WhatsAppContextStore


CHAT_ID = "120363001234567890@g.us"
BOT_ID = "15551230000@s.whatsapp.net"
JACOB = "209066827718687@lid"
COUNTERPARTY = "5216641112222@s.whatsapp.net"


def _store(tmp_path: Path) -> WhatsAppContextStore:
    return WhatsAppContextStore(tmp_path / "wa-context")


def _msg(message_id: str, body: str, *, sender=JACOB, sender_name="Jacob", **extra):
    data = {
        "chatId": CHAT_ID,
        "messageId": message_id,
        "senderId": sender,
        "senderName": sender_name,
        "timestamp": "2026-06-21T18:00:00Z",
        "body": body,
        "isGroup": True,
        "mentionedIds": [],
        "botIds": [BOT_ID],
    }
    data.update(extra)
    return data


def test_contract_pdf_is_stored_silently_and_used_when_later_mentioned(tmp_path: Path):
    store = _store(tmp_path)
    pdf_path = tmp_path / "contrato.pdf"
    pdf_path.write_text("CONTRATO DE PRESTACION DE SERVICIOS\nClausula primera: pago y diligencia.", encoding="utf-8")

    store.ingest_bridge_message(
        _msg(
            "m1",
            "[document received: contrato.pdf]",
            hasMedia=True,
            mediaType="document",
            mediaFileName="contrato.pdf",
            mediaUrls=[str(pdf_path)],
            extractedText="Contrato: pago realizado; queda agendar diligencia.",
        )
    )

    trigger = store.ingest_bridge_message(
        _msg(
            "m2",
            "@Jack dame resumen corto del contrato",
            mentionedIds=[BOT_ID],
        )
    )
    bundle = store.build_context_bundle(CHAT_ID, trigger.message_id)
    rendered = bundle.render_for_prompt()

    assert "m1" in bundle.context_message_ids
    assert str(pdf_path) in bundle.media_paths
    assert "Contrato: pago realizado" in rendered
    assert "Reenvíalo" not in rendered


def test_payment_proof_and_counterparty_confirmation_advance_next_step(tmp_path: Path):
    store = _store(tmp_path)
    proof_path = tmp_path / "pago 9.5k ger.pdf"
    proof_path.write_text("Comprobante SPEI por $9,500 a Gestión Inmobiliaria y Legal. Mercantil 247/2023.", encoding="utf-8")

    store.ingest_bridge_message(
        _msg(
            "m1",
            "[document received: pago 9.5k ger.pdf]",
            hasMedia=True,
            mediaType="document",
            mediaFileName="pago 9.5k ger.pdf",
            mediaUrls=[str(proof_path)],
            extractedText="SPEI $9,500 MXN beneficiario Gestión Inmobiliaria y Legal expediente Mercantil 247/2023",
        )
    )
    store.ingest_bridge_message(
        _msg(
            "m2",
            "Confirmo de recibido",
            sender=COUNTERPARTY,
            sender_name="Gestión Inmobiliaria Y Legal",
        )
    )
    trigger = store.ingest_bridge_message(
        _msg(
            "m3",
            "Jack me recuerdas aqui cual es el siguiente paso con @Gestión Inmobiliaria Y Legal ? Ya revisaste si hay nuevos acuerdos o actualizaciones en los casos de Eduardo?",
            mentionedIds=[BOT_ID],
        )
    )

    bundle = store.build_context_bundle(CHAT_ID, trigger.message_id)
    rendered = bundle.render_for_prompt()

    assert bundle.task_state["payment_proof_sent"] is True
    assert bundle.task_state["receipt_confirmed_by_counterparty"] is True
    assert bundle.guardrails["block_payment_recommendation"] is True
    assert "pago 9.5k ger.pdf" in rendered
    assert "$9,500" in rendered or "9500" in rendered
    assert "Mercantil 247/2023" in rendered
    assert "Do not recommend paying, authorizing payment, or sending the comprobante" in rendered
    assert "request date/time of diligence" in rendered
    assert "request official receipt" in rendered


def test_download_failure_is_explicit_and_does_not_pretend_to_read_pdf(tmp_path: Path):
    store = _store(tmp_path)
    store.ingest_bridge_message(
        _msg(
            "m1",
            "[document received: contrato.pdf]",
            hasMedia=True,
            mediaType="document",
            mediaFileName="contrato.pdf",
            mediaUrls=[],
            mediaDownloadStatus="failed",
        )
    )
    trigger = store.ingest_bridge_message(
        _msg("m2", "@Jack resumelo", mentionedIds=[BOT_ID])
    )

    bundle = store.build_context_bundle(CHAT_ID, trigger.message_id)
    rendered = bundle.render_for_prompt()

    assert bundle.unreadable_media
    assert "Veo que se envió un archivo, pero no pude descargarlo o leerlo. Reenvíalo o respóndeme directamente al archivo." in rendered
    assert "pretend" in rendered.lower()


def test_mention_only_stores_unmentioned_messages_silently_then_uses_them(tmp_path: Path):
    store = _store(tmp_path)

    first = store.ingest_bridge_message(_msg("m1", "Eduardo: hablaron de Mercantil 247/2023"))
    second = store.ingest_bridge_message(_msg("m2", "El pago ya quedó enviado"))
    trigger = store.ingest_bridge_message(
        _msg("m3", "@Jack siguiente paso?", mentionedIds=[BOT_ID])
    )

    assert first.should_reply is False
    assert second.should_reply is False
    assert trigger.should_reply is True

    bundle = store.build_context_bundle(CHAT_ID, trigger.message_id)
    assert {"m1", "m2", "m3"}.issubset(set(bundle.context_message_ids))
    rendered = bundle.render_for_prompt()
    assert "Eduardo" in rendered
    assert "Mercantil 247/2023" in rendered
