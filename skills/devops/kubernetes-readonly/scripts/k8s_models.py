"""Strict Pydantic models for read-only kubectl requests (bundled skill)."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from k8s_validators import check_name, check_ns, check_resource


class OpVersion(BaseModel):
    """kubectl version -o json (client + server when reachable)."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["version"]


class OpClusterInfo(BaseModel):
    """kubectl cluster-info (read-only discovery)."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["cluster_info"]


class OpApiResources(BaseModel):
    """kubectl api-resources."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["api_resources"]
    api_group: Annotated[str, StringConstraints(max_length=253)] | None = None

    @field_validator("api_group")
    @classmethod
    def _grp(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", v):
            raise ValueError("invalid api_group")
        return v


class OpExplain(BaseModel):
    """kubectl explain RESOURCE."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["explain"]
    resource: str = Field(..., min_length=1, max_length=253)
    recursive: bool = False

    @field_validator("resource")
    @classmethod
    def _res(cls, v: str) -> str:
        return check_resource(v)


class OpGet(BaseModel):
    """kubectl get (JSON/YAML/wide/name only)."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["get"]
    resource: str
    name: str | None = None
    namespace: str | None = None
    all_namespaces: bool = False
    output: Literal["json", "yaml", "wide", "name", "default"] = "json"

    @model_validator(mode="after")
    def _ns_scope(self) -> OpGet:
        if self.all_namespaces and self.namespace:
            raise ValueError("namespace is incompatible with all_namespaces")
        return self

    @field_validator("resource")
    @classmethod
    def _res(cls, v: str) -> str:
        return check_resource(v)

    @field_validator("name")
    @classmethod
    def _nm(cls, v: str | None) -> str | None:
        return check_name(v)

    @field_validator("namespace")
    @classmethod
    def _ns(cls, v: str | None) -> str | None:
        return check_ns(v)


class OpDescribe(BaseModel):
    """kubectl describe."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["describe"]
    resource: str
    name: str
    namespace: str | None = None

    @field_validator("resource")
    @classmethod
    def _res(cls, v: str) -> str:
        return check_resource(v)

    @field_validator("name")
    @classmethod
    def _nm(cls, v: str) -> str:
        checked = check_name(v)
        if not checked:
            raise ValueError("name is required")
        return checked

    @field_validator("namespace")
    @classmethod
    def _ns(cls, v: str | None) -> str | None:
        return check_ns(v)


class OpTopPods(BaseModel):
    """kubectl top pods (metrics-server)."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["top_pods"]
    namespace: str | None = None
    all_namespaces: bool = False

    @model_validator(mode="after")
    def _ns_scope(self) -> OpTopPods:
        if self.all_namespaces and self.namespace:
            raise ValueError("namespace is incompatible with all_namespaces")
        return self

    @field_validator("namespace")
    @classmethod
    def _ns(cls, v: str | None) -> str | None:
        return check_ns(v)


class OpTopNodes(BaseModel):
    """kubectl top nodes."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["top_nodes"]


K8sRequest = Union[
    OpVersion,
    OpClusterInfo,
    OpApiResources,
    OpExplain,
    OpGet,
    OpDescribe,
    OpTopPods,
    OpTopNodes,
]
