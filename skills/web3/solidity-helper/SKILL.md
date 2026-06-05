---
name: solidity-helper
description: Yerel olarak akıllı sözleşme güvenlik denetimi yapar ve güvenli ERC-20 / ERC-721 taslakları üretir.
version: 1.0.0
author: Ibrahim Uylas
license: MIT
metadata:
  hermes:
    tags: [web3, blockchain, solidity, security]
    category: development
---

# Solidity Helper Skill

## When to Use
Kullanıcı bir akıllı sözleşme (Solidity) yazmak istediğinde, mevcut bir sözleşmenin güvenlik açıklarını denetlemek istediğinde veya standart bir ERC-20 / ERC-721 taslağına ihtiyaç duyduğunda bu skill'i kullan.

## Procedure
1. Kullanıcı kod veya güvenlik analizi talep ederse, mevcut dosya okuma yeteneklerini kullan veya kodu doğrudan analiz et.
2. Güvenlik denetimi yapılacaksa şu başlıkları kontrol et:
   - Reentrancy (yeniden giriş hatası)
   - Integer overflow / underflow
   - Eksik erişim kontrolü (access control)
3. Bulguları net bir şekilde listele.
4. Yeni sözleşme taslağı gerekirse OpenZeppelin standartlarına uygun güvenli Solidity kodu üret.

## Security Constraints
- Dış kaynaklardan veri çekme.
- Sadece kullanıcının verdiği kod ve yerel bilgiyi kullan.
- Bu analizin otomatik bir yardımcı olduğunu, mutlaka denetime tabi tutulması gerektiğini belirt.

## Verification
Çıktıdaki kodlar güncel Solidity sürümü ile başlamalı ve markdown olarak düzgün biçimlendirilmelidir.
