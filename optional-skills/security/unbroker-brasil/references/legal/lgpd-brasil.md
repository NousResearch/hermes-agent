# LGPD Brazil legal reference for unbroker-brasil

This is an operational reference, not legal advice.

## Core rights

Use the Lei Geral de Proteção de Dados (LGPD, Lei nº 13.709/2018) as the default legal frame for Brazilian data-subject removal requests.

Useful hooks:

- Art. 6: purpose, adequacy, necessity, free access, data quality, transparency, security, prevention, non-discrimination, accountability.
- Art. 15: termination of personal-data processing.
- Art. 16: deletion after processing terminates, subject to legal retention exceptions.
- Art. 18: data-subject rights, including confirmation, access, correction, anonymization, blocking, elimination, portability, information about sharing, and revocation.
- Art. 41: controller should indicate the Encarregado/DPO communication channel.

## Standard request language

Ask for:

1. Confirmation whether the controller processes the subject's data.
2. Immediate removal, anonymization, or blocking of excessive/unnecessary/stale personal data.
3. De-indexing/noindex where the controller publishes public pages.
4. List of third parties with whom the data was shared, when applicable.
5. Confirmation of completion by e-mail.

## Minimal disclosure

Default fields:

- Name.
- Contact e-mail for response.
- URL(s) of exposed page(s).
- City/state only when required to identify the record.
- Phone only if the page already exposes it or the source requires it to locate the record.

Do not automatically disclose:

- CPF.
- RG.
- CNH.
- Birth date.
- Selfie.
- Utility bill.
- Signature image.
- Copies of documents.

If a channel requires any of those, queue `human_task_queued` and ask the user once in the final digest.

## Public records

LGPD does not mean every public record can be erased. For official court, notary, property, Receita, or government records:

- Do not promise deletion.
- Request mirror suppression/de-indexing when the mirror republishes unnecessary data.
- Prepare a human/legal task if official correction is needed.

## Escalation package

For non-response or refusal, prepare:

- Subject identity fields used in the request.
- Source URL(s).
- Date/time of request.
- Recipient/channel.
- Copy of message body.
- Response or lack of response.
- Screenshot or text evidence of continued exposure.

Escalation options:

1. Encarregado/DPO escalation.
2. Platform/hosting abuse/privacy channel.
3. Search engine removal/de-indexing.
4. ANPD complaint package.
5. User/legal counsel for official records or identity-proof demands.
