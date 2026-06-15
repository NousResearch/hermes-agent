"""Standard service abstraction — 層級：Core Logic Layer (Layer 2)

所有 Service 須對齊此抽象接口，核心調度層僅面向抽象接口溝通。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, FrozenSet


class ServiceInterface(ABC):
    """標準服務抽象基類"""

    @abstractmethod
    def health_check(self) -> bool:
        """服務健康檢查"""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """取得服務配置"""
        ...


class LLMServiceInterface(ServiceInterface):
    """LLM 調用抽象接口"""

    @abstractmethod
    def chat_completion(
        self,
        messages: list,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """標準 chat completion 調用"""
        ...

    @abstractmethod
    def classify_error(self, error: Exception) -> str:
        """錯誤分類：retryable / fatal / rate_limit"""
        ...

    @abstractmethod
    def get_retry_delay(self, attempt: int, error_type: str) -> float:
        """重試退避策略（對齊 CODEX.md 重試策略表）"""
        ...


class ToolServiceInterface(ServiceInterface):
    """工具服務抽象接口 — 對齊 BaseToolService"""

    @abstractmethod
    def discover_tools(self, agent) -> FrozenSet:
        """返回代理可調用的工具名稱集合"""
        ...

    @abstractmethod
    def validate_args(self, function_name: str, function_args: dict) -> Any:
        """驗證並清理工具參數"""
        ...

    @abstractmethod
    def format_output(self, result: Any, function_name: str) -> dict[str, Any]:
        """格式化工具執行結果用於 transcript 注入"""
        ...

    @abstractmethod
    async def apply_middleware(
        self,
        agent,
        request: Any,
    ) -> tuple[dict, list[dict]]:
        """應用工具請求中介層，返回 (修改後參數, 追蹤事件)"""
        ...

    @abstractmethod
    def enforce_budget(self, messages: list, task_id: str) -> None:
        """強制執行每輪聚合預算，超限則拋出 BudgetExceededError"""
        ...

    @abstractmethod
    async def execute(
        self,
        agent,
        request: Any,
    ) -> Any:
        """執行工具的完整業務邏輯管道"""
        ...
