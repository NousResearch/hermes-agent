"""RecruitmentSystem AI query support."""

from .models import QueryRequest, QueryResponse, TimeRange, UserContext
from .service import query_recruitment_system

__all__ = [
    "QueryRequest",
    "QueryResponse",
    "TimeRange",
    "UserContext",
    "query_recruitment_system",
]
