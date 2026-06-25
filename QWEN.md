<!-- HERMES_OWNER_RULES_START -->
> **Hermes Owner Rules (ใช้ทุกโปรเจกต์)**
>
> กฎนี้อยู่เหนือกฎภาษาเดิมในไฟล์โปรเจกต์ ถ้าขัดกันให้ใช้กฎนี้ก่อน

## กติกากลางที่ AI ต้องทำตาม

1. ใช้ภาษาของผู้ใช้ก่อนเสมอ ถ้าผู้ใช้พิมพ์ไทย ให้ตอบไทยทั้งคำอธิบาย สรุป ความเสี่ยง และขั้นตอนถัดไป
2. ถ้าจำเป็นต้องใช้ศัพท์เทคนิค ให้แปลเป็นภาษาคนทันที เช่น `registry` = สมุดทะเบียนติดตาม, `traceability` = ร่องรอยว่าเอาไปใช้ที่ไหนแล้ว, `adapter` = ไฟล์เชื่อมบริบทให้ AI อ่าน
3. ถ้าผู้ใช้ส่งลิงก์ บทความ โพสต์ วิดีโอ หรือข้อมูลให้เรียนรู้ ให้สรุปและเสนอทางเลือกในแชทก่อน แล้วรอเจ้าของงานเลือกก่อนบันทึกลงไฟล์ ความจำถาวร สมุดทะเบียนติดตาม หรือระบบความรู้ เว้นแต่เจ้าของงานสั่งชัดว่าให้เลือกและทำได้เลยในรอบนั้น
4. ถ้าต้องแก้หลายไฟล์หรือหลายเฟส ให้ทำงานเป็นระบบ แยกเฟส ตรวจงานจริง และรายงานผลเป็นภาษาคนที่เจ้าของงานตัดสินใจต่อได้
5. ถ้าผู้ใช้เรียก Shortcut เช่น `Use Act-As`, `Use Comply`, `Use Summary`, `Use Scan Feature`, `Use Opus Plan`, `Use AI Pair`, `Use Pair AI`, `Use Business Plan`, `Use Viber Structure`, `Use Viber Audit`, `Use Impeccable`, `Use Blog Auto`, `Use WOW Resource`, `Use Flow Guardian`, `Use New Chat`, `Use Save Git`, `Use Continue`, `Review Chat` หรือชื่อย่อที่ใกล้เคียง ต้องเปิดไฟล์ทะเบียน Shortcut และ Prompt เต็มก่อนใช้ ห้ามเดาหรือใช้จากความจำ
6. ห้ามสรุปงานด้วยภาษาเครื่องมืออย่างเดียว เช่น `parse passed`, `promoted`, `registry updated` ต้องบอกความหมายและผลกระทบเป็นภาษาไทยด้วย

## ไฟล์ความจำกลางที่ต้องอ้างอิง

- `/Users/rattanasak/ObsidianVault/HermesAgent/AI_MEMORY.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/global-context.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/prompt-shortcut-registry.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/memory/profile/user-language-first.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/memory/profile/knowledge-intake-review-before-write.md`

ถ้ากำลังทำงานบน VPS ให้ใช้ mirror root `/home/linux-nat/ObsidianVault/HermesAgent` แทน `/Users/rattanasak/ObsidianVault/HermesAgent` สำหรับไฟล์ความจำกลางชุดเดียวกัน

อย่าโหลดทั้ง vault ถ้าไม่จำเป็น อ่านไฟล์กลางข้างบนก่อน แล้วค่อยค้นเพิ่มเฉพาะเมื่อข้อมูลไม่พอ
<!-- HERMES_OWNER_RULES_END -->

# Qwen Project Memory - Hermes Agent

## Obsidian Context Bridge

Use this file as a lightweight Qwen adapter. The source of truth is Obsidian.

Read before substantial work:

- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/session-start-contract.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/global-context.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/prompt-shortcut-registry.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/99-System/context-packs/hermes-agent-dev.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/projects/hermes-agent-dev/project-context.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/projects/hermes-agent-dev/active-memory.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/projects/hermes-agent-dev/handoff.md`

On VPS, replace the local Obsidian root with `/home/linux-nat/ObsidianVault/HermesAgent`.

Use repo-local `.hermes/context.md`, `.hermes/active.md`, and `.hermes/decisions.md` when available.

Do not duplicate long Obsidian knowledge here. This file points Qwen to the right context.

## Prompt Shortcuts

When the user invokes `Use Act-As`, `Use Comply`, `Use Summary`, `Use Scan Feature`, `Use Opus Plan`, `Use AI Pair`, `Use Pair AI`, `Use Business Plan`, `Use Viber Structure`, `Use Viber Audit`, `Use Impeccable`, `Use Blog Auto`, `Use WOW Resource`, `Use Flow Guardian`, `Use New Chat`, `Use Save Git`, `Use Continue`, `Review Chat`, or an alias, read `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/prompt-shortcut-registry.md` and then open the mapped prompt file. Do not guess or summarize the shortcut from memory.
