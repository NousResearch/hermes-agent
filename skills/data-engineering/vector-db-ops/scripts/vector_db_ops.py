#!/usr/bin/env python3
"""Unified vector database operations for Hermes Agent."""
import argparse
import os
import sys
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Try imports
CLIENTS = {}

def import_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
        CLIENTS["qdrant"] = (QdrantClient, Distance, VectorParams, Filter, FieldCondition, MatchValue)
        return True
    except ImportError:
        return False

def import_pinecone():
    try:
        from pinecone import Pinecone, ServerlessSpec
        CLIENTS["pinecone"] = (Pinecone, ServerlessSpec)
        return True
    except ImportError:
        return False

def import_weaviate():
    try:
        import weaviate
        import weaviate.classes as wvc
        CLIENTS["weaviate"] = (weaviate, wvc)
        return True
    except ImportError:
        return False

def import_milvus():
    try:
        from pymilvus import MilvusClient, DataType
        CLIENTS["milvus"] = (MilvusClient, DataType)
        return True
    except ImportError:
        return False

def import_chroma():
    try:
        import chromadb
        CLIENTS["chroma"] = chromadb
        return True
    except ImportError:
        return False


class VectorDB(ABC):
    @abstractmethod
    def create_collection(self, name: str, dimension: int, distance: str) -> None: pass
    
    @abstractmethod
    def list_collections(self) -> List[str]: pass
    
    @abstractmethod
    def describe_collection(self, name: str) -> Dict: pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None: pass
    
    @abstractmethod
    def upsert(self, collection: str, vectors: List[Dict]) -> None: pass
    
    @abstractmethod
    def search(self, collection: str, vector: List[float], limit: int, filter_dict: Optional[Dict], with_payload: bool, with_vectors: bool) -> List[Dict]: pass
    
    @abstractmethod
    def backup(self, collection: str, output_dir: str) -> None: pass
    
    @abstractmethod
    def restore(self, collection: str, input_dir: str) -> None: pass


class QdrantDB(VectorDB):
    def __init__(self):
        if not import_qdrant():
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
        QdrantClient, Distance, VectorParams, Filter, FieldCondition, MatchValue = CLIENTS["qdrant"]
        url = os.environ.get("VECTOR_DB_URL", "http://localhost:6333")
        api_key = os.environ.get("VECTOR_DB_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.Distance = Distance
        self.VectorParams = VectorParams
        self.Filter = Filter
        self.FieldCondition = FieldCondition
        self.MatchValue = MatchValue
    
    def _distance(self, d: str):
        return {"cosine": self.Distance.COSINE, "euclid": self.Distance.EUCLID, "dot": self.Distance.DOT}.get(d, self.Distance.COSINE)
    
    def _build_filter(self, filter_dict: Dict):
        if not filter_dict:
            return None
        conditions = []
        for key, value in filter_dict.items():
            conditions.append(self.FieldCondition(key=key, match=self.MatchValue(value=value)))
        return self.Filter(must=conditions)
    
    def create_collection(self, name: str, dimension: int, distance: str):
        self.client.create_collection(
            collection_name=name,
            vectors_config=self.VectorParams(size=dimension, distance=self._distance(distance))
        )
    
    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.get_collections().collections]
    
    def describe_collection(self, name: str) -> Dict:
        info = self.client.get_collection(name)
        return {"name": name, "vectors_count": info.vectors_count, "points_count": info.points_count, "status": info.status}
    
    def delete_collection(self, name: str):
        self.client.delete_collection(name)
    
    def upsert(self, collection: str, vectors: List[Dict]):
        points = []
        for v in vectors:
            points.append({
                "id": v["id"],
                "vector": v["vector"],
                "payload": v.get("payload", {})
            })
        self.client.upsert(collection_name=collection, points=points)
    
    def search(self, collection: str, vector: List[float], limit: int, filter_dict: Optional[Dict], with_payload: bool, with_vectors: bool) -> List[Dict]:
        results = self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            query_filter=self._build_filter(filter_dict),
            with_payload=with_payload,
            with_vectors=with_vectors
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload, "vector": r.vector} for r in results]
    
    def backup(self, collection: str, output_dir: str):
        # Qdrant doesn't have native backup via API, snapshot via CLI
        raise NotImplementedError("Qdrant backup requires CLI snapshot command")
    
    def restore(self, collection: str, input_dir: str):
        raise NotImplementedError("Qdrant restore requires CLI snapshot command")


class PineconeDB(VectorDB):
    def __init__(self):
        if not import_pinecone():
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")
        Pinecone, ServerlessSpec = CLIENTS["pinecone"]
        api_key = os.environ.get("VECTOR_DB_API_KEY")
        if not api_key:
            raise ValueError("VECTOR_DB_API_KEY required for Pinecone")
        self.client = Pinecone(api_key=api_key)
        self.ServerlessSpec = ServerlessSpec
    
    def _metric(self, d: str):
        return {"cosine": "cosine", "euclid": "euclidean", "dot": "dotproduct"}.get(d, "cosine")
    
    def create_collection(self, name: str, dimension: int, distance: str):
        self.client.create_index(
            name=name,
            dimension=dimension,
            metric=self._metric(distance),
            spec=self.ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    def list_collections(self) -> List[str]:
        return [idx.name for idx in self.client.list_indexes()]
    
    def describe_collection(self, name: str) -> Dict:
        desc = self.client.describe_index(name)
        return {"name": name, "dimension": desc.dimension, "metric": desc.metric, "status": desc.status}
    
    def delete_collection(self, name: str):
        self.client.delete_index(name)
    
    def upsert(self, collection: str, vectors: List[Dict]):
        index = self.client.Index(collection)
        to_upsert = [(v["id"], v["vector"], v.get("payload", {})) for v in vectors]
        index.upsert(vectors=to_upsert)
    
    def search(self, collection: str, vector: List[float], limit: int, filter_dict: Optional[Dict], with_payload: bool, with_vectors: bool) -> List[Dict]:
        index = self.client.Index(collection)
        filter_expr = None
        if filter_dict:
            # Pinecone filter syntax
            filter_expr = {k: {"$eq": v} for k, v in filter_dict.items()}
        results = index.query(
            vector=vector,
            top_k=limit,
            filter=filter_expr,
            include_metadata=with_payload,
            include_values=with_vectors
        )
        return [{"id": m.id, "score": m.score, "payload": m.metadata, "vector": m.values} for m in results.matches]
    
    def backup(self, collection: str, output_dir: str):
        raise NotImplementedError("Pinecone backup not implemented")
    
    def restore(self, collection: str, input_dir: str):
        raise NotImplementedError("Pinecone restore not implemented")


class WeaviateDB(VectorDB):
    def __init__(self):
        if not import_weaviate():
            raise ImportError("weaviate-client not installed. Run: pip install weaviate-client")
        weaviate, wvc = CLIENTS["weaviate"]
        url = os.environ.get("VECTOR_DB_URL", "http://localhost:8080")
        api_key = os.environ.get("VECTOR_DB_API_KEY")
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        self.client = weaviate.connect_to_local(host=url.replace("http://", "").replace("https://", ""), auth_credentials=auth)
        self.wvc = wvc
    
    def create_collection(self, name: str, dimension: int, distance: str):
        self.client.collections.create(
            name=name,
            vectorizer_config=self.wvc.config.Configure.Vectorizer.none(),
            vector_index_config=self.wvc.config.Configure.VectorIndex.hnsw(distance_metric=self.wvc.config.VectorDistances.COSINE)
        )
    
    def list_collections(self) -> List[str]:
        return list(self.client.collections.list_all().keys())
    
    def describe_collection(self, name: str) -> Dict:
        coll = self.client.collections.get(name)
        return {"name": name, "config": str(coll.config.get())}
    
    def delete_collection(self, name: str):
        self.client.collections.delete(name)
    
    def upsert(self, collection: str, vectors: List[Dict]):
        coll = self.client.collections.get(collection)
        with coll.batch.dynamic() as batch:
            for v in vectors:
                batch.add_object(properties=v.get("payload", {}), vector=v["vector"], uuid=v["id"])
    
    def search(self, collection: str, vector: List[float], limit: int, filter_dict: Optional[Dict], with_payload: bool, with_vectors: bool) -> List[Dict]:
        coll = self.client.collections.get(collection)
        where = None
        if filter_dict:
            filters = []
            for k, v in filter_dict.items():
                filters.append(self.wvc.query.Filter.by_property(k).equal(v))
            where = self.wvc.query.Filter.all_of(filters) if len(filters) > 1 else filters[0]
        
        results = coll.query.near_vector(
            near_vector=vector,
            limit=limit,
            where=where,
            return_metadata=self.wvc.query.MetadataQuery(score=True),
            return_properties=list(v.get("payload", {}).keys()) if with_payload else None,
            include_vector=with_vectors
        )
        return [{"id": str(o.uuid), "score": o.metadata.score, "payload": o.properties, "vector": o.vector} for o in results.objects]
    
    def backup(self, collection: str, output_dir: str):
        raise NotImplementedError("Weaviate backup not implemented")
    
    def restore(self, collection: str, input_dir: str):
        raise NotImplementedError("Weaviate restore not implemented")


class MilvusDB(VectorDB):
    def __init__(self):
        if not import_milvus():
            raise ImportError("pymilvus not installed. Run: pip install pymilvus")
        MilvusClient, DataType = CLIENTS["milvus"]
        url = os.environ.get("VECTOR_DB_URL", "http://localhost:19530")
        self.client = MilvusClient(uri=url)
    
    def create_collection(self, name: str, dimension: int, distance: str):
        metric = {"cosine": "COSINE", "euclid": "L2", "dot": "IP"}.get(distance, "COSINE")
        self.client.create_collection(
            collection_name=name,
            dimension=dimension,
            metric_type=metric,
            consistency_level="Strong"
        )
    
    def list_collections(self) -> List[str]:
        return self.client.list_collections()
    
    def describe_collection(self, name: str) -> Dict:
        return self.client.describe_collection(name)
    
    def delete_collection(self, name: str):
        self.client.drop_collection(name)
    
    def upsert(self, collection: str, vectors: List[Dict]):
        data = []
        for v in vectors:
            item = {"id": v["id"], "vector": v["vector"]}
            item.update(v.get("payload", {}))
            data.append(item)
        self.client.insert(collection_name=collection, data=data)
    
    def search(self, collection: str, vector: List[float], limit: int, filter_dict: Optional[Dict], with_payload: bool, with_vectors: bool) -> List[Dict]:
        filter_expr = None
        if filter_dict:
            # Milvus filter syntax
            conditions = [f'{k} == "{v}"' for k, v in filter_dict.items()]
            filter_expr = " and ".join(conditions)
        
        results = self.client.search(
            collection_name=collection,
            data=[vector],
            limit=limit,
            filter=filter_expr,
            output_fields=["*"] if with_payload else None
        )
        out = []
        for hits in results:
            for hit in hits:
                item = {"id": hit["id"], "score": hit["distance"]}
                if with_payload:
                    item["payload"] = {k: v for k, v in hit.items() if k not in ["id", "distance", "vector"]}
                if with_vectors and "vector" in hit:
                    item["vector"] = hit["vector"]
                out.append(item)
        return out
    
    def backup(self, collection: str, output_dir: str):
        raise NotImplementedError("Milvus backup not implemented")
    
    def restore(self, collection: str, input_dir: str):
        raise NotImplementedError("Milvus restore not implemented")


class ChromaDB(VectorDB):
    def __init__(self):
        if not import_chroma():
            raise ImportError("chromadb not installed. Run: pip install chromadb")
        chromadb = CLIENTS["chroma"]
        url = os.environ.get("VECTOR_DB_URL", "http://localhost:8000")
        if url.startswith("http"):
            # HTTP client
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            self.client = chromadb.HttpClient(host=parsed.hostname, port=parsed.port or 8000)
        else:
            # Local persistent client
            self.client = chromadb.PersistentClient(path=url)
    
    def create_collection(self, name: str, dimension: int, distance: str):
        self.client.create_collection(name=name, metadata={"hnsw:space": {"cosine": "cosine", "euclid": "l2", "dot": "ip"}.get(distance, "cosine")})
    
    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]
    
    def describe_collection(self, name: str) -> Dict:
        coll = self.client.get_collection(name)
        return {"name": name, "count": coll.count()}
    
    def delete_collection(self, name: str):
        self.client.delete_collection(name)
    
    def upsert(self, collection: str, vectors: List[Dict]):
        coll = self.client.get_collection(collection)
        ids = [v["id"] for v in vectors]
        embeddings = [v["vector"] for v in vectors]
        metadatas = [v.get("payload", {}) for v in vectors]
        coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    
    def search(self, collection: str, vector: List[float], limit: int, filter_dict: Optional[Dict], with_payload: bool, with_vectors: bool) -> List[Dict]:
        coll = self.client.get_collection(collection)
        where = filter_dict if filter_dict else None
        results = coll.query(
            query_embeddings=[vector],
            n_results=limit,
            where=where,
            include=["metadatas", "documents", "distances", "embeddings"] if with_payload else ["distances"]
        )
        out = []
        if results["ids"]:
            for i, id_val in enumerate(results["ids"][0]):
                item = {"id": id_val, "score": results["distances"][0][i]}
                if with_payload:
                    item["payload"] = results["metadatas"][0][i] if results["metadatas"] else {}
                if with_vectors and results["embeddings"]:
                    item["vector"] = results["embeddings"][0][i]
                out.append(item)
        return out
    
    def backup(self, collection: str, output_dir: str):
        # Chroma supports export
        raise NotImplementedError("Chroma backup not implemented")
    
    def restore(self, collection: str, input_dir: str):
        raise NotImplementedError("Chroma restore not implemented")


def get_db() -> VectorDB:
    db_type = os.environ.get("VECTOR_DB_TYPE", "").lower()
    if db_type == "qdrant":
        return QdrantDB()
    elif db_type == "pinecone":
        return PineconeDB()
    elif db_type == "weaviate":
        return WeaviateDB()
    elif db_type == "milvus":
        return MilvusDB()
    elif db_type == "chroma":
        return ChromaDB()
    else:
        raise ValueError(f"Unsupported VECTOR_DB_TYPE: {db_type}. Supported: qdrant, pinecone, weaviate, milvus, chroma")


def cmd_collection(args: argparse.Namespace):
    db = get_db()
    if args.action == "create":
        db.create_collection(args.name, args.dimension, args.distance)
        print(f"Collection '{args.name}' created")
    elif args.action == "list":
        collections = db.list_collections()
        if args.format == "json":
            json.dump(collections, sys.stdout)
            sys.stdout.write("\n")
        else:
            for c in collections:
                print(c)
    elif args.action == "describe":
        info = db.describe_collection(args.name)
        if args.format == "json":
            json.dump(info, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            for k, v in info.items():
                print(f"{k}: {v}")
    elif args.action == "delete":
        db.delete_collection(args.name)
        print(f"Collection '{args.name}' deleted")


def cmd_upsert(args: argparse.Namespace):
    db = get_db()
    vectors = []
    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line:
                    vectors.append(json.loads(line))
    elif args.id and args.vector:
        vector = json.loads(args.vector) if isinstance(args.vector, str) else args.vector
        payload = json.loads(args.payload) if args.payload else {}
        vectors.append({"id": args.id, "vector": vector, "payload": payload})
    else:
        raise ValueError("Either --file or (--id and --vector) required")
    
    db.upsert(args.collection, vectors)
    print(f"Upserted {len(vectors)} vectors to '{args.collection}'")


def cmd_search(args: argparse.Namespace):
    db = get_db()
    vector = json.loads(args.vector) if isinstance(args.vector, str) else args.vector
    filter_dict = json.loads(args.filter) if args.filter else None
    
    results = db.search(
        collection=args.collection,
        vector=vector,
        limit=args.limit,
        filter_dict=filter_dict,
        with_payload=args.with_payload,
        with_vectors=args.with_vectors
    )
    
    if args.format == "json":
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            print(f"ID: {r['id']}, Score: {r['score']:.4f}")
            if r.get("payload"):
                print(f"  Payload: {json.dumps(r['payload'], ensure_ascii=False)}")
            if r.get("vector"):
                print(f"  Vector: [{len(r['vector'])} dims]")
            print()


def cmd_backup(args: argparse.Namespace):
    db = get_db()
    db.backup(args.collection, args.output)
    print(f"Backup of '{args.collection}' saved to '{args.output}'")


def cmd_restore(args: argparse.Namespace):
    db = get_db()
    db.restore(args.collection, args.input)
    print(f"Restored '{args.collection}' from '{args.input}'")


def main():
    parser = argparse.ArgumentParser(description="Unified vector database operations")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # collection command
    p_coll = subparsers.add_parser("collection", help="Collection management")
    p_coll.add_argument("--action", choices=["create", "list", "describe", "delete"], required=True)
    p_coll.add_argument("--name", help="Collection name")
    p_coll.add_argument("--dimension", type=int, help="Vector dimension (for create)")
    p_coll.add_argument("--distance", choices=["cosine", "euclid", "dot"], default="cosine", help="Distance metric")
    
    # upsert command
    p_upsert = subparsers.add_parser("upsert", help="Upsert vectors")
    p_upsert.add_argument("--collection", required=True)
    p_upsert.add_argument("--file", help="JSONL file with vectors")
    p_upsert.add_argument("--id", help="Single vector ID")
    p_upsert.add_argument("--vector", help="Single vector as JSON array")
    p_upsert.add_argument("--payload", help="Single vector payload as JSON")
    
    # search command
    p_search = subparsers.add_parser("search", help="ANN search")
    p_search.add_argument("--collection", required=True)
    p_search.add_argument("--vector", required=True, help="Query vector as JSON array")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--filter", help="Filter as JSON object")
    p_search.add_argument("--with-payload", action="store_true", default=True)
    p_search.add_argument("--with-vectors", action="store_true", default=False)
    
    # backup command
    p_backup = subparsers.add_parser("backup", help="Backup collection")
    p_backup.add_argument("--collection", required=True)
    p_backup.add_argument("--output", required=True)
    
    # restore command
    p_restore = subparsers.add_parser("restore", help="Restore collection")
    p_restore.add_argument("--collection", required=True)
    p_restore.add_argument("--input", required=True)
    
    args = parser.parse_args()
    
    if args.command == "collection":
        cmd_collection(args)
    elif args.command == "upsert":
        cmd_upsert(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "backup":
        cmd_backup(args)
    elif args.command == "restore":
        cmd_restore(args)


if __name__ == "__main__":
    main()