# Architecture

`HermesSQLiteCollector`는 profile별 DB를 범위 쿼리하고 `StateStore`의 durable buffer에
넣는다. `EpisodeDetector`는 이 buffer를 시간순으로 분리해 quiet 상태인 회의만
`HegiPipeline`에 전달한다.

`HierarchicalMeetingAnalyzer`는 발언 경계를 지키며 긴 Episode를 chunk 요약한 뒤
최종 JSON 회의록을 생성한다. `build_minutes`는 source message ID를 실제 Episode와
대조하고 근거 없는 Action Item을 제외한다. `MemoryEvaluator`는 read MCP 검색 결과와
회의의 지속성을 비교한다.

`ArchiveManager`가 local spool에 Markdown/JSON을 atomic write한 뒤 NAS가 접근 가능한
경우 복제한다. `TelegramReporter`는 네 개의 보고 part를 각각 checkpoint하여 부분
실패 뒤 이미 성공한 part를 재전송하지 않는다. 성공한 실제 보고 뒤에만 buffer를
consumed로 전환한다.
